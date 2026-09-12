import { spawn } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { sepilotdHome } from '../../../storage/home.js'
import type { MediaOperation, Provider } from '../adapter.js'

const LOCAL_PROVIDER_ID = 'local-python'
const DEFAULT_TIMEOUT_MS = 30 * 60_000
const DEFAULT_INSTALL_TIMEOUT_MS = 20 * 60_000
const PACKAGE_MARKER = '.sepilotd-imagegen-packages-v1'
const GENERATOR_SCRIPT = 'generate_image.py'
const OPERATIONS: MediaOperation[] = [
  'text-to-image',
  'image-to-image',
  'inpaint',
  'text-to-video',
  'image-to-video',
]
const DEFAULT_MODELS: Record<MediaOperation, string> = {
  'text-to-image': 'stabilityai/sd-turbo',
  'image-to-image': 'timbrooks/instruct-pix2pix',
  inpaint: 'runwayml/stable-diffusion-inpainting',
  'text-to-video': 'cerspense/zeroscope_v2_576w',
  'image-to-video': 'stabilityai/stable-video-diffusion-img2vid-xt',
}
const MODEL_ENV: Record<MediaOperation, string> = {
  'text-to-image': 'SEPILOTD_IMAGE_GEN_MODEL',
  'image-to-image': 'SEPILOTD_IMAGE_EDIT_MODEL',
  inpaint: 'SEPILOTD_IMAGE_INPAINT_MODEL',
  'text-to-video': 'SEPILOTD_TEXT_VIDEO_MODEL',
  'image-to-video': 'SEPILOTD_IMAGE_VIDEO_MODEL',
}

interface LocalPythonParams {
  operation?: MediaOperation
  model?: string
  modelId?: string
  variant?: string
  workspace?: string
  venvDir?: string
  outputDir?: string
  width?: number
  height?: number
  count?: number
  batchSize?: number
  seed?: number
  negativePrompt?: string
  imagePath?: string
  imageDataUrl?: string
  maskPath?: string
  maskDataUrl?: string
  strength?: number
  frames?: number
  numFrames?: number
  fps?: number
  outputFormat?: 'mp4' | 'gif'
  lowVram?: boolean
  disableSafetyChecker?: boolean
  allowCpu?: boolean
  device?: 'auto' | 'cpu' | 'cuda' | 'mps'
  gpuIndex?: number
  pipeline?: string
  dtype?: 'auto' | 'float16' | 'bfloat16' | 'float32'
  loraPath?: string
  loraWeightName?: string
  loraScale?: number
  scheduler?: string
  schedulerTimestepSpacing?: string
  trueCfgScale?: number
  imageGuidanceScale?: number
  motionBucketId?: number
  noiseAugStrength?: number
  decodeChunkSize?: number
  steps?: number
  guidanceScale?: number
  cfgScale?: number
  maxSequenceLength?: number
  useSafetensors?: boolean
  timeoutMs?: number
  installTimeoutMs?: number
  forceInstall?: boolean
  python?: string
  torchIndexUrl?: string
}

interface GeneratedImageMetadata {
  outputs: Array<{
    path: string
    seed?: number
    width?: number
    height?: number
    mime?: string
    kind?: 'image' | 'video'
  }>
  model: string
  device: string
  operation?: MediaOperation
  elapsedSec?: number
}

export interface LocalPythonModelCacheStatus {
  modelId: string
  cached: boolean
  cachePath?: string
  dependencies?: Array<{
    kind: 'lora'
    modelId: string
    cached: boolean
    cachePath?: string
  }>
}

export type LocalPythonEnvironmentIssueCode =
  | 'python_unavailable'
  | 'venv_module_unavailable'
  | 'venv_broken'
  | 'venv_pip_unavailable'
  | 'venv_packages_broken'

export interface LocalPythonEnvironmentStatus {
  providerId: typeof LOCAL_PROVIDER_ID
  ready: boolean
  code?: LocalPythonEnvironmentIssueCode
  message: string
  python: {
    ok: boolean
    command: string
    version?: string
    error?: string
  }
  venv: {
    ok: boolean
    path: string
    pythonPath: string
    state: 'missing' | 'needs-install' | 'ready' | 'broken'
    packagesInstalled: boolean
    error?: string
  }
}

function asObject(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function stringParam(
  params: Record<string, unknown>,
  key: keyof LocalPythonParams,
): string | undefined {
  const value = params[key]
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function numberParam(
  params: Record<string, unknown>,
  key: keyof LocalPythonParams,
): number | undefined {
  const value = params[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function boolParam(
  params: Record<string, unknown>,
  key: keyof LocalPythonParams,
): boolean | undefined {
  const value = params[key]
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase()
    if (['1', 'true', 'yes', 'on'].includes(lowered)) return true
    if (['0', 'false', 'no', 'off'].includes(lowered)) return false
  }
  return undefined
}

function operationParam(params: Record<string, unknown>): MediaOperation {
  const raw = stringParam(params, 'operation') ?? 'text-to-image'
  if (!OPERATIONS.includes(raw as MediaOperation)) {
    throw new Error(`params.operation must be one of ${OPERATIONS.join(', ')}`)
  }
  return raw as MediaOperation
}

function parseParams(
  raw: unknown,
): Required<Pick<LocalPythonParams, 'device'>> & LocalPythonParams {
  const params = asObject(raw)
  const operation = operationParam(params)
  const model =
    stringParam(params, 'model') ??
    stringParam(params, 'modelId') ??
    process.env[MODEL_ENV[operation]] ??
    (operation !== 'text-to-image' ? process.env.SEPILOTD_IMAGE_GEN_MODEL : undefined) ??
    DEFAULT_MODELS[operation]

  const device = stringParam(params, 'device') ?? 'auto'
  if (!['auto', 'cpu', 'cuda', 'mps'].includes(device)) {
    throw new Error('params.device must be one of auto, cpu, cuda, or mps')
  }
  const dtype = stringParam(params, 'dtype') ?? 'auto'
  if (!['auto', 'float16', 'bfloat16', 'float32'].includes(dtype)) {
    throw new Error('params.dtype must be one of auto, float16, bfloat16, or float32')
  }
  const gpuIndex = numberParam(params, 'gpuIndex')
  if (gpuIndex !== undefined && (!Number.isInteger(gpuIndex) || gpuIndex < 0)) {
    throw new Error('params.gpuIndex must be a non-negative integer')
  }

  return {
    operation,
    model,
    modelId: stringParam(params, 'modelId'),
    variant: stringParam(params, 'variant'),
    workspace: stringParam(params, 'workspace'),
    venvDir: stringParam(params, 'venvDir'),
    outputDir: stringParam(params, 'outputDir'),
    width: numberParam(params, 'width'),
    height: numberParam(params, 'height'),
    count: numberParam(params, 'count'),
    batchSize: numberParam(params, 'batchSize'),
    seed: numberParam(params, 'seed'),
    negativePrompt: stringParam(params, 'negativePrompt'),
    imagePath: stringParam(params, 'imagePath'),
    imageDataUrl: stringParam(params, 'imageDataUrl'),
    maskPath: stringParam(params, 'maskPath'),
    maskDataUrl: stringParam(params, 'maskDataUrl'),
    strength: numberParam(params, 'strength'),
    frames: numberParam(params, 'frames'),
    numFrames: numberParam(params, 'numFrames'),
    fps: numberParam(params, 'fps'),
    outputFormat: stringParam(params, 'outputFormat') === 'gif' ? 'gif' : 'mp4',
    lowVram: boolParam(params, 'lowVram'),
    disableSafetyChecker: boolParam(params, 'disableSafetyChecker'),
    allowCpu: boolParam(params, 'allowCpu'),
    device: device as 'auto' | 'cpu' | 'cuda' | 'mps',
    gpuIndex,
    pipeline: stringParam(params, 'pipeline'),
    dtype: dtype as 'auto' | 'float16' | 'bfloat16' | 'float32',
    loraPath: stringParam(params, 'loraPath'),
    loraWeightName: stringParam(params, 'loraWeightName'),
    loraScale: numberParam(params, 'loraScale'),
    scheduler: stringParam(params, 'scheduler'),
    schedulerTimestepSpacing: stringParam(params, 'schedulerTimestepSpacing'),
    trueCfgScale: numberParam(params, 'trueCfgScale'),
    imageGuidanceScale: numberParam(params, 'imageGuidanceScale'),
    motionBucketId: numberParam(params, 'motionBucketId'),
    noiseAugStrength: numberParam(params, 'noiseAugStrength'),
    decodeChunkSize: numberParam(params, 'decodeChunkSize'),
    steps: numberParam(params, 'steps'),
    guidanceScale: numberParam(params, 'guidanceScale') ?? numberParam(params, 'cfgScale'),
    cfgScale: numberParam(params, 'cfgScale'),
    maxSequenceLength: numberParam(params, 'maxSequenceLength'),
    useSafetensors: boolParam(params, 'useSafetensors'),
    timeoutMs: numberParam(params, 'timeoutMs'),
    installTimeoutMs: numberParam(params, 'installTimeoutMs'),
    forceInstall: boolParam(params, 'forceInstall'),
    python: stringParam(params, 'python'),
    torchIndexUrl:
      stringParam(params, 'torchIndexUrl') ?? process.env.SEPILOTD_IMAGE_GEN_TORCH_INDEX_URL,
  }
}

function defaultVenvDir(params: LocalPythonParams): string {
  if (params.venvDir) return resolve(params.venvDir)
  if (params.workspace && isAbsolute(params.workspace)) {
    return join(params.workspace, '.sepilotd-imagegen', 'venv')
  }
  return join(sepilotdHome(), 'image-gen', 'local', 'venv')
}

function defaultWorkDir(params: LocalPythonParams): string {
  if (params.outputDir) return resolve(params.outputDir)
  if (params.workspace && isAbsolute(params.workspace)) {
    return join(params.workspace, '.sepilotd-imagegen', 'outputs')
  }
  return join(sepilotdHome(), 'image-gen', 'local', 'outputs')
}

function hfHubCacheDir(): string {
  if (process.env.HUGGINGFACE_HUB_CACHE) return resolve(process.env.HUGGINGFACE_HUB_CACHE)
  if (process.env.HF_HUB_CACHE) return resolve(process.env.HF_HUB_CACHE)
  if (process.env.TRANSFORMERS_CACHE) return resolve(process.env.TRANSFORMERS_CACHE)
  const hfHome = process.env.HF_HOME
    ? resolve(process.env.HF_HOME)
    : join(homedir(), '.cache', 'huggingface')
  return join(hfHome, 'hub')
}

function isRemoteModelRef(value: string): boolean {
  if (!value.trim()) return false
  if (isAbsolute(value)) return false
  if (value.startsWith('.') || value.includes('\\')) return false
  return !existsSync(value)
}

function modelCacheStatus(modelId: string): { cached: boolean; cachePath?: string } {
  if (!isRemoteModelRef(modelId)) {
    const path = resolve(modelId)
    return { cached: existsSync(path), cachePath: path }
  }
  const path = join(hfHubCacheDir(), `models--${modelId.replaceAll('/', '--')}`)
  if (!existsSync(path)) return { cached: false, cachePath: path }
  const snapshots = join(path, 'snapshots')
  try {
    const hasSnapshot = existsSync(snapshots) && readdirSync(snapshots).length > 0
    return { cached: hasSnapshot, cachePath: path }
  } catch {
    return { cached: false, cachePath: path }
  }
}

export function localPythonModelCacheStatuses(): Record<string, LocalPythonModelCacheStatus> {
  const statuses: Record<string, LocalPythonModelCacheStatus> = {}
  for (const model of localPythonProvider.info.recommendedModels ?? []) {
    const modelId = model.modelId ?? model.id
    const primary = modelCacheStatus(modelId)
    const loraPath = typeof model.params?.loraPath === 'string' ? model.params.loraPath : undefined
    const dependencies = loraPath
      ? [
          {
            kind: 'lora' as const,
            modelId: loraPath,
            ...modelCacheStatus(loraPath),
          },
        ]
      : undefined
    statuses[model.id] = {
      modelId,
      ...primary,
      ...(dependencies ? { dependencies } : {}),
    }
  }
  return statuses
}

function venvPython(venvDir: string): string {
  return process.platform === 'win32'
    ? join(venvDir, 'Scripts', 'python.exe')
    : join(venvDir, 'bin', 'python')
}

function tail(text: string, max = 4000): string {
  return text.length <= max ? text : text.slice(text.length - max)
}

function runProcess(
  command: string,
  args: string[],
  options: {
    cwd?: string
    env?: NodeJS.ProcessEnv
    timeoutMs: number
    signal?: AbortSignal
  },
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    let settled = false

    const finish = (error?: Error, result?: { stdout: string; stderr: string }) => {
      if (settled) return
      settled = true
      clearTimeout(timeout)
      options.signal?.removeEventListener('abort', onAbort)
      if (error) reject(error)
      else resolvePromise(result ?? { stdout, stderr })
    }

    const onAbort = () => {
      child.kill('SIGTERM')
      finish(new Error('image generation cancelled'))
    }

    const timeout = setTimeout(() => {
      child.kill('SIGTERM')
      finish(
        new Error(`command timed out after ${options.timeoutMs}ms: ${command} ${args.join(' ')}`),
      )
    }, options.timeoutMs)

    options.signal?.addEventListener('abort', onAbort, { once: true })
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk) => {
      stdout += chunk
    })
    child.stderr.on('data', (chunk) => {
      stderr += chunk
    })
    child.on('error', (error) => finish(error))
    child.on('close', (code) => {
      if (code === 0) {
        finish(undefined, { stdout, stderr })
        return
      }
      finish(
        new Error(
          `command failed (${code}): ${command} ${args.join(' ')}\n${tail(stderr || stdout)}`,
        ),
      )
    })
  })
}

async function ensureVenv(
  params: LocalPythonParams,
  venvDir: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const python = venvPython(venvDir)
  const installTimeoutMs = params.installTimeoutMs ?? DEFAULT_INSTALL_TIMEOUT_MS
  if (!existsSync(python)) {
    await mkdir(dirname(venvDir), { recursive: true })
    const defaultBasePython = process.platform === 'win32' ? 'python' : 'python3'
    const basePython = params.python ?? process.env.SEPILOTD_IMAGE_GEN_PYTHON ?? defaultBasePython
    await runProcess(basePython, ['-m', 'venv', venvDir], { timeoutMs: installTimeoutMs, signal })
  }

  const marker = join(venvDir, PACKAGE_MARKER)
  if (!existsSync(marker) || params.forceInstall) {
    await runProcess(python, ['-m', 'pip', 'install', '--upgrade', 'pip'], {
      timeoutMs: installTimeoutMs,
      signal,
    })
    const torchArgs = ['-m', 'pip', 'install', 'torch']
    if (params.torchIndexUrl) torchArgs.push('--index-url', params.torchIndexUrl)
    await runProcess(python, torchArgs, {
      timeoutMs: installTimeoutMs,
      signal,
    })
    await runProcess(
      python,
      [
        '-m',
        'pip',
        'install',
        'diffusers',
        'transformers',
        'accelerate',
        'safetensors',
        'pillow',
        'peft',
        'imageio',
        'imageio-ffmpeg',
      ],
      { timeoutMs: installTimeoutMs, signal },
    )
    await writeFile(marker, new Date().toISOString(), 'utf-8')
  }
  return python
}

const ENVIRONMENT_PROBE_TIMEOUT_MS = 30_000
const IMAGE_GEN_IMPORT_PROBE = [
  'from PIL import Image',
  'import torch',
  'from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image, DiffusionPipeline',
  'from diffusers.utils import export_to_gif, export_to_video',
  'print("ok")',
].join('; ')

function firstOutputLine(output: { stdout: string; stderr: string }): string | undefined {
  const line = `${output.stdout}\n${output.stderr}`
    .split(/\r?\n/)
    .map((item) => item.trim())
    .find(Boolean)
  return line || undefined
}

async function probeCommand(
  command: string,
  args: string[],
): Promise<{ ok: true; output?: string } | { ok: false; error: string }> {
  try {
    const output = await runProcess(command, args, { timeoutMs: ENVIRONMENT_PROBE_TIMEOUT_MS })
    return { ok: true, output: firstOutputLine(output) }
  } catch (error) {
    return {
      ok: false,
      error: tail(error instanceof Error ? error.message : String(error), 1000),
    }
  }
}

function basePythonCommand(params: LocalPythonParams = {}): string {
  const defaultBasePython = process.platform === 'win32' ? 'python' : 'python3'
  return params.python ?? process.env.SEPILOTD_IMAGE_GEN_PYTHON ?? defaultBasePython
}

export async function localPythonEnvironmentStatus(): Promise<LocalPythonEnvironmentStatus> {
  const params: LocalPythonParams = {}
  const venvDir = defaultVenvDir(params)
  const pythonPath = venvPython(venvDir)
  const command = basePythonCommand(params)

  const python = await probeCommand(command, ['--version'])
  if (!python.ok) {
    return {
      providerId: LOCAL_PROVIDER_ID,
      ready: false,
      code: 'python_unavailable',
      message: '로컬 Python 실행 파일을 찾거나 실행할 수 없습니다.',
      python: { ok: false, command, error: python.error },
      venv: {
        ok: false,
        path: venvDir,
        pythonPath,
        state: existsSync(venvDir) ? 'broken' : 'missing',
        packagesInstalled: false,
      },
    }
  }

  const venvModule = await probeCommand(command, ['-m', 'venv', '--help'])
  if (!venvModule.ok) {
    return {
      providerId: LOCAL_PROVIDER_ID,
      ready: false,
      code: 'venv_module_unavailable',
      message: 'Python venv 모듈을 사용할 수 없습니다.',
      python: { ok: true, command, version: python.output },
      venv: {
        ok: false,
        path: venvDir,
        pythonPath,
        state: existsSync(venvDir) ? 'broken' : 'missing',
        packagesInstalled: false,
        error: venvModule.error,
      },
    }
  }

  const venvExists = existsSync(venvDir)
  const venvPythonExists = existsSync(pythonPath)
  if (!venvPythonExists) {
    const brokenExistingVenv = venvExists
    return {
      providerId: LOCAL_PROVIDER_ID,
      ready: !brokenExistingVenv,
      code: brokenExistingVenv ? 'venv_broken' : undefined,
      message: brokenExistingVenv
        ? '기존 이미지 생성 venv에서 Python 실행 파일을 찾을 수 없습니다.'
        : 'Python과 venv 모듈을 확인했습니다. 첫 생성 시 격리된 venv를 만듭니다.',
      python: { ok: true, command, version: python.output },
      venv: {
        ok: !brokenExistingVenv,
        path: venvDir,
        pythonPath,
        state: brokenExistingVenv ? 'broken' : 'missing',
        packagesInstalled: false,
      },
    }
  }

  const venvPythonProbe = await probeCommand(pythonPath, ['--version'])
  if (!venvPythonProbe.ok) {
    return {
      providerId: LOCAL_PROVIDER_ID,
      ready: false,
      code: 'venv_broken',
      message: '이미지 생성 venv의 Python을 실행할 수 없습니다.',
      python: { ok: true, command, version: python.output },
      venv: {
        ok: false,
        path: venvDir,
        pythonPath,
        state: 'broken',
        packagesInstalled: existsSync(join(venvDir, PACKAGE_MARKER)),
        error: venvPythonProbe.error,
      },
    }
  }

  const marker = join(venvDir, PACKAGE_MARKER)
  const packagesInstalled = existsSync(marker)
  if (!packagesInstalled) {
    const pip = await probeCommand(pythonPath, ['-m', 'pip', '--version'])
    if (!pip.ok) {
      return {
        providerId: LOCAL_PROVIDER_ID,
        ready: false,
        code: 'venv_pip_unavailable',
        message: '이미지 생성 venv에서 pip를 사용할 수 없습니다.',
        python: { ok: true, command, version: python.output },
        venv: {
          ok: false,
          path: venvDir,
          pythonPath,
          state: 'broken',
          packagesInstalled,
          error: pip.error,
        },
      }
    }
    return {
      providerId: LOCAL_PROVIDER_ID,
      ready: true,
      message: '이미지 생성 venv를 확인했습니다. 첫 생성 시 필요한 패키지를 설치합니다.',
      python: { ok: true, command, version: python.output },
      venv: {
        ok: true,
        path: venvDir,
        pythonPath,
        state: 'needs-install',
        packagesInstalled,
      },
    }
  }

  const packageProbe = await probeCommand(pythonPath, ['-c', IMAGE_GEN_IMPORT_PROBE])
  if (!packageProbe.ok) {
    return {
      providerId: LOCAL_PROVIDER_ID,
      ready: false,
      code: 'venv_packages_broken',
      message: '이미지 생성 venv의 필수 Python 패키지를 불러올 수 없습니다.',
      python: { ok: true, command, version: python.output },
      venv: {
        ok: false,
        path: venvDir,
        pythonPath,
        state: 'broken',
        packagesInstalled,
        error: packageProbe.error,
      },
    }
  }

  return {
    providerId: LOCAL_PROVIDER_ID,
    ready: true,
    message: '로컬 Python 이미지 생성 환경이 준비되었습니다.',
    python: { ok: true, command, version: python.output },
    venv: {
      ok: true,
      path: venvDir,
      pythonPath,
      state: 'ready',
      packagesInstalled,
    },
  }
}

const SCRIPT = String.raw`import base64
import inspect
import io
import json
import os
import time
from pathlib import Path

from PIL import Image


def load_input_image(spec, path_key, data_key, label):
    path = spec.get(path_key)
    data_url = spec.get(data_key)
    if path:
        return Image.open(path).convert("RGB")
    if data_url:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")
    raise RuntimeError(f"{label} is required for operation {spec.get('operation')}")


def resize_if_requested(image, width, height):
    if width and height and image.size != (width, height):
        return image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def call_pipe(pipe, kwargs):
    signature = inspect.signature(pipe.__call__)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if accepts_kwargs:
        return pipe(**kwargs)
    return pipe(**{k: v for k, v in kwargs.items() if k in signature.parameters})


def output_frames(result):
    frames = getattr(result, "frames", None)
    if frames is None and isinstance(result, dict):
        frames = result.get("frames")
    if frames is None:
        raise RuntimeError("diffusers pipeline did not return video frames")
    if frames and isinstance(frames, list) and frames and isinstance(frames[0], list):
        return frames[0]
    return frames


def main():
    with open(os.environ["SEPILOTD_IMAGE_GEN_INPUT"], "r", encoding="utf-8") as handle:
        spec = json.load(handle)

    import torch
    from diffusers import (
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        AutoPipelineForText2Image,
        DiffusionPipeline,
    )
    from diffusers.utils import export_to_gif, export_to_video

    requested_device = spec.get("device", "auto")
    allow_cpu = bool(spec.get("allowCpu", False))
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device

    if device == "cpu" and not allow_cpu:
        raise RuntimeError("CPU execution is disabled; pass params.allowCpu=true to allow a CPU run")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device == "mps" and (getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")

    operation = spec.get("operation", "text-to-image")
    model_id = spec["model"]
    pipeline = (spec.get("pipeline") or "auto").lower()
    model_id_lower = model_id.lower()
    if pipeline == "auto" and "qwen-image-edit" in model_id_lower:
        pipeline = "qwen-image-edit"

    dtype_name = spec.get("dtype") or "auto"
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
    load_kwargs = {"torch_dtype": dtype}
    if spec.get("variant"):
        load_kwargs["variant"] = spec["variant"]
    if spec.get("useSafetensors"):
        load_kwargs["use_safetensors"] = True

    scheduler_name = (spec.get("scheduler") or "").lower()
    if scheduler_name == "qwen-lightning-flowmatch-euler":
        import math
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        load_kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    if pipeline in ("diffusion-pipeline", "generic", "qwen-image", "qwen-image-edit"):
        pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "text-to-video-sd":
        try:
            from diffusers import TextToVideoSDPipeline
        except ImportError as exc:
            raise RuntimeError("Text-to-video requires a diffusers build with TextToVideoSDPipeline") from exc
        pipe = TextToVideoSDPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "stable-video-diffusion":
        try:
            from diffusers import StableVideoDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Stable Video Diffusion requires a diffusers build with StableVideoDiffusionPipeline") from exc
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "qwen-image-edit-plus":
        try:
            from diffusers import QwenImageEditPlusPipeline
            pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, **load_kwargs)
        except ImportError:
            pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "kolors":
        try:
            from diffusers import KolorsPipeline
            pipe = KolorsPipeline.from_pretrained(model_id, **load_kwargs)
        except ImportError:
            pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "kolors-img2img":
        try:
            from diffusers import KolorsImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError("Kolors img2img requires a diffusers build with KolorsImg2ImgPipeline") from exc
        pipe = KolorsImg2ImgPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "instruct-pix2pix":
        try:
            from diffusers import StableDiffusionInstructPix2PixPipeline
        except ImportError as exc:
            raise RuntimeError("InstructPix2Pix requires a diffusers build with StableDiffusionInstructPix2PixPipeline") from exc
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "qwen-image-edit-inpaint":
        try:
            from diffusers import QwenImageEditInpaintPipeline
        except ImportError as exc:
            raise RuntimeError("Qwen Image Edit inpaint requires a diffusers build with QwenImageEditInpaintPipeline") from exc
        pipe = QwenImageEditInpaintPipeline.from_pretrained(model_id, **load_kwargs)
    elif pipeline == "sana":
        try:
            from diffusers import SanaPipeline
        except ImportError as exc:
            raise RuntimeError("Sana requires a diffusers build with SanaPipeline") from exc
        sana_load_kwargs = dict(load_kwargs)
        if dtype_name in ("float16", "bfloat16"):
            sana_load_kwargs["torch_dtype"] = torch.float32
        pipe = SanaPipeline.from_pretrained(model_id, **sana_load_kwargs)
        if dtype_name == "bfloat16":
            if hasattr(pipe, "text_encoder"):
                pipe.text_encoder.to(torch.bfloat16)
            if hasattr(pipe, "transformer"):
                pipe.transformer = pipe.transformer.to(torch.bfloat16)
        elif dtype_name == "float16" and hasattr(pipe, "transformer"):
            pipe.transformer = pipe.transformer.to(torch.float16)
    elif operation == "text-to-image":
        pipe = AutoPipelineForText2Image.from_pretrained(spec["model"], **load_kwargs)
    elif operation == "image-to-image":
        pipe = AutoPipelineForImage2Image.from_pretrained(spec["model"], **load_kwargs)
    elif operation == "inpaint":
        pipe = AutoPipelineForInpainting.from_pretrained(spec["model"], **load_kwargs)
    else:
        pipe = DiffusionPipeline.from_pretrained(spec["model"], **load_kwargs)

    if spec.get("disableSafetyChecker") and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    if scheduler_name and scheduler_name not in ("default", "qwen-lightning-flowmatch-euler"):
        if scheduler_name == "lcm":
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif scheduler_name == "euler":
            from diffusers import EulerDiscreteScheduler
            scheduler_kwargs = {}
            if spec.get("schedulerTimestepSpacing"):
                scheduler_kwargs["timestep_spacing"] = spec["schedulerTimestepSpacing"]
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_kwargs)
        else:
            raise RuntimeError(f"Unsupported scheduler: {scheduler_name}")

    if spec.get("loraPath"):
        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError("The selected pipeline does not support LoRA loading")
        lora_kwargs = {}
        if spec.get("loraWeightName"):
            lora_kwargs["weight_name"] = spec["loraWeightName"]
        pipe.load_lora_weights(spec["loraPath"], **lora_kwargs)
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=float(spec.get("loraScale") or 1.0))

    if spec.get("lowVram") and device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    output_dir = Path(spec["outputDir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    count = max(1, min(int(spec.get("count") or 1), 8))
    width = int(spec.get("width") or 512)
    height = int(spec.get("height") or 512)
    steps = int(spec.get("steps") or 20)
    guidance_scale = float(spec["guidanceScale"]) if spec.get("guidanceScale") is not None else 7.5
    true_cfg_scale = spec.get("trueCfgScale")
    image_guidance_scale = (
        float(spec["imageGuidanceScale"]) if spec.get("imageGuidanceScale") is not None else None
    )
    motion_bucket_id = spec.get("motionBucketId")
    noise_aug_strength = spec.get("noiseAugStrength")
    decode_chunk_size = spec.get("decodeChunkSize")
    strength = float(spec.get("strength") or 0.65)
    if operation in ("image-to-image", "inpaint"):
        strength = max(0.01, min(strength, 1.0))
        if int(steps * strength) < 1:
            strength = min(1.0, 1.0 / max(steps, 1))
    frames = max(1, min(int(spec.get("numFrames") or 16), 64))
    fps = max(1, min(int(spec.get("fps") or 8), 30))
    output_format = spec.get("outputFormat") or "mp4"
    seed = spec.get("seed")
    if seed is not None and int(seed) < 0:
        seed = None
    negative_prompt = spec.get("negativePrompt") or None
    started = time.time()
    outputs = []

    if operation in ("image-to-image", "inpaint", "image-to-video"):
        image = load_input_image(spec, "imagePath", "imageDataUrl", "input image")
        image = resize_if_requested(image, width, height)
    else:
        image = None
    if operation == "inpaint":
        mask_image = load_input_image(spec, "maskPath", "maskDataUrl", "mask image")
        mask_image = resize_if_requested(mask_image, width, height)
    else:
        mask_image = None

    if operation == "text-to-video":
        item_seed = int(seed) if seed is not None else None
        generator = (
            torch.Generator(device="cpu").manual_seed(item_seed)
            if item_seed is not None
            else None
        )
        kwargs = {
            "prompt": spec["prompt"],
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if true_cfg_scale is not None:
            kwargs["true_cfg_scale"] = float(true_cfg_scale)
        if spec.get("maxSequenceLength"):
            kwargs["max_sequence_length"] = int(spec["maxSequenceLength"])
        result = call_pipe(pipe, kwargs)
        frames_out = output_frames(result)
        if output_format == "gif":
            path = output_dir / f"{spec['jobId']}-video-0.gif"
            export_to_gif(frames_out, str(path))
            mime = "image/gif"
        else:
            path = output_dir / f"{spec['jobId']}-video-0.mp4"
            export_to_video(frames_out, str(path), fps=fps)
            mime = "video/mp4"
        outputs.append({
            "path": str(path),
            "seed": item_seed,
            "width": width,
            "height": height,
            "mime": mime,
            "kind": "video",
        })
    elif operation == "image-to-video":
        item_seed = int(seed) if seed is not None else None
        generator = (
            torch.Generator(device="cpu").manual_seed(item_seed)
            if item_seed is not None
            else None
        )
        if pipeline == "stable-video-diffusion":
            kwargs = {
                "image": image,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if motion_bucket_id is not None:
                kwargs["motion_bucket_id"] = int(motion_bucket_id)
            if noise_aug_strength is not None:
                kwargs["noise_aug_strength"] = float(noise_aug_strength)
            if decode_chunk_size is not None:
                kwargs["decode_chunk_size"] = int(decode_chunk_size)
        else:
            kwargs = {
                "prompt": spec["prompt"],
                "negative_prompt": negative_prompt,
                "image": image,
                "height": height,
                "width": width,
                "num_frames": frames,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if true_cfg_scale is not None:
                kwargs["true_cfg_scale"] = float(true_cfg_scale)
            if spec.get("maxSequenceLength"):
                kwargs["max_sequence_length"] = int(spec["maxSequenceLength"])
        result = call_pipe(pipe, kwargs)
        frames_out = output_frames(result)
        if output_format == "gif":
            path = output_dir / f"{spec['jobId']}-video-0.gif"
            export_to_gif(frames_out, str(path))
            mime = "image/gif"
        else:
            path = output_dir / f"{spec['jobId']}-video-0.mp4"
            export_to_video(frames_out, str(path), fps=fps)
            mime = "video/mp4"
        outputs.append({
            "path": str(path),
            "seed": item_seed,
            "width": width,
            "height": height,
            "mime": mime,
            "kind": "video",
        })
    else:
        for index in range(count):
            generator = None
            item_seed = None
            if seed is not None:
                item_seed = int(seed) + index
                generator = torch.Generator(device="cpu").manual_seed(item_seed)
            kwargs = {
                "prompt": spec["prompt"],
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if true_cfg_scale is not None:
                kwargs["true_cfg_scale"] = float(true_cfg_scale)
            if spec.get("maxSequenceLength"):
                kwargs["max_sequence_length"] = int(spec["maxSequenceLength"])
            if operation == "image-to-image":
                kwargs["image"] = [image] if pipeline == "qwen-image-edit-plus" else image
                kwargs["strength"] = strength
                if image_guidance_scale is not None:
                    kwargs["image_guidance_scale"] = image_guidance_scale
            if operation == "inpaint":
                kwargs["image"] = image
                kwargs["mask_image"] = mask_image
                kwargs["strength"] = strength
            result = call_pipe(pipe, kwargs)
            image_out = result.images[0]
            if not isinstance(image_out, Image.Image):
                raise RuntimeError("diffusers pipeline did not return a PIL image")
            path = output_dir / f"{spec['jobId']}-{index}.png"
            image_out.save(path)
            with Image.open(path) as verify:
                verify.verify()
            outputs.append({
                "path": str(path),
                "seed": item_seed,
                "width": width,
                "height": height,
                "mime": "image/png",
                "kind": "image",
            })

    print(json.dumps({
        "outputs": outputs,
        "model": spec["model"],
        "device": device,
        "operation": operation,
        "elapsedSec": round(time.time() - started, 3),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
`

async function writeGeneratorScript(workDir: string): Promise<string> {
  const scriptPath = join(workDir, GENERATOR_SCRIPT)
  await mkdir(workDir, { recursive: true })
  await writeFile(scriptPath, SCRIPT, 'utf-8')
  return scriptPath
}

function parseMetadata(stdout: string): GeneratedImageMetadata {
  const line = stdout
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean)
    .pop()
  if (!line) throw new Error('image generator produced no JSON metadata')
  return JSON.parse(line) as GeneratedImageMetadata
}

export const localPythonProvider: Provider = {
  info: {
    id: LOCAL_PROVIDER_ID,
    label: 'Local Python (diffusers)',
    enabled: true,
    operations: OPERATIONS,
    hardware: {
      supportsDeviceSelection: true,
      devicesEndpoint: '/image-gen/hardware',
    },
    recommendedModels: [
      {
        id: DEFAULT_MODELS['text-to-image'],
        label: 'SD Turbo',
        operation: 'text-to-image',
        notes: 'Fast text-to-image default for low VRAM GPUs.',
        minVramMiB: 4096,
        recommendedVramMiB: 6144,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['fast', 'sd15'],
        workflow: {
          id: 'diffusers-auto-t2i',
          label: 'Diffusers AutoPipeline',
          description: 'Loads the model through AutoPipelineForText2Image.',
        },
        params: { steps: 2, guidanceScale: 0 },
      },
      {
        id: 'stabilityai/sdxl-turbo',
        label: 'SDXL Turbo',
        operation: 'text-to-image',
        notes: 'Higher quality turbo preset for 8GB+ GPUs.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['fast', 'sdxl'],
        workflow: {
          id: 'diffusers-auto-t2i',
          label: 'Diffusers AutoPipeline',
          description: 'Loads the model through AutoPipelineForText2Image.',
        },
        params: { steps: 4, guidanceScale: 0 },
      },
      {
        id: 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        label: 'Stable Diffusion 1.5',
        operation: 'text-to-image',
        notes: 'Classic low VRAM baseline with broad LoRA ecosystem support.',
        minVramMiB: 4096,
        recommendedVramMiB: 6144,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['lightweight', 'sd15'],
        workflow: {
          id: 'diffusers-auto-t2i',
          label: 'Diffusers AutoPipeline',
          description: 'Loads the model through AutoPipelineForText2Image.',
        },
        params: { dtype: 'float16', steps: 25, guidanceScale: 7.5 },
      },
      {
        id: 'Lykon/DreamShaper',
        label: 'DreamShaper',
        operation: 'text-to-image',
        notes: 'Popular SD 1.5 finetune for illustrative, fantasy, and semi-realistic output.',
        minVramMiB: 4096,
        recommendedVramMiB: 6144,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['lightweight', 'sd15', 'art'],
        workflow: {
          id: 'diffusers-auto-t2i',
          label: 'Diffusers AutoPipeline',
          description: 'Loads the model through AutoPipelineForText2Image.',
        },
        params: { dtype: 'float16', steps: 25, guidanceScale: 7 },
      },
      {
        id: 'stabilityai/stable-diffusion-2',
        label: 'Stable Diffusion 2',
        operation: 'text-to-image',
        notes: 'Moderate VRAM 768px Stability baseline.',
        minVramMiB: 6144,
        recommendedVramMiB: 8192,
        deviceKinds: ['cuda', 'mps'],
        tags: ['stable-diffusion', '768px'],
        workflow: {
          id: 'diffusers-auto-t2i',
          label: 'Diffusers AutoPipeline',
          description: 'Loads the model through AutoPipelineForText2Image.',
        },
        params: { dtype: 'float16', steps: 30, guidanceScale: 7.5 },
      },
      {
        id: 'stabilityai/stable-diffusion-xl-base-1.0',
        label: 'SDXL Base 1.0',
        operation: 'text-to-image',
        notes: 'General purpose SDXL baseline with broad ecosystem compatibility.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['sdxl', 'baseline'],
        workflow: {
          id: 'diffusers-auto-t2i',
          label: 'Diffusers AutoPipeline',
          description: 'Loads the model through AutoPipelineForText2Image.',
        },
        params: {
          variant: 'fp16',
          dtype: 'float16',
          useSafetensors: true,
          steps: 30,
          guidanceScale: 7,
        },
      },
      {
        id: 'sdxl-lightning-4step-lora',
        modelId: 'stabilityai/stable-diffusion-xl-base-1.0',
        label: 'SDXL Lightning 4-step',
        operation: 'text-to-image',
        notes: 'Fast SDXL workflow using the ByteDance SDXL-Lightning LoRA.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['fast', 'sdxl', 'lora', 'workflow'],
        workflow: {
          id: 'sdxl-lightning-lora',
          label: 'SDXL + Lightning LoRA',
          description:
            'Loads SDXL base, applies the 4-step Lightning LoRA, and uses Euler trailing.',
        },
        params: {
          variant: 'fp16',
          dtype: 'float16',
          useSafetensors: true,
          loraPath: 'ByteDance/SDXL-Lightning',
          loraWeightName: 'sdxl_lightning_4step_lora.safetensors',
          scheduler: 'euler',
          schedulerTimestepSpacing: 'trailing',
          steps: 4,
          guidanceScale: 0,
        },
      },
      {
        id: 'sdxl-lcm-lora-4step',
        modelId: 'stabilityai/stable-diffusion-xl-base-1.0',
        label: 'SDXL LCM LoRA 4-step',
        operation: 'text-to-image',
        notes: 'Low-step SDXL workflow using the Latent Consistency LoRA.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['fast', 'sdxl', 'lora', 'workflow'],
        workflow: {
          id: 'sdxl-lcm-lora',
          label: 'SDXL + LCM LoRA',
          description: 'Loads SDXL base, applies LCM LoRA, and switches to LCMScheduler.',
        },
        params: {
          variant: 'fp16',
          dtype: 'float16',
          useSafetensors: true,
          loraPath: 'latent-consistency/lcm-lora-sdxl',
          scheduler: 'lcm',
          steps: 4,
          guidanceScale: 1.5,
        },
      },
      {
        id: 'segmind/SSD-1B',
        label: 'Segmind SSD-1B',
        operation: 'text-to-image',
        notes: 'Distilled SDXL-style model that targets faster/lighter 1024px generation.',
        minVramMiB: 6144,
        recommendedVramMiB: 8192,
        deviceKinds: ['cuda', 'mps'],
        tags: ['lightweight', 'sdxl'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 45,
          guidanceScale: 7,
        },
      },
      {
        id: 'playgroundai/playground-v2.5-1024px-aesthetic',
        label: 'Playground v2.5 1024px',
        operation: 'text-to-image',
        notes: 'Aesthetic SDXL-class model for polished 1024px images.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['high-quality', 'sdxl', '1024px'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          variant: 'fp16',
          dtype: 'float16',
          lowVram: true,
          steps: 50,
          guidanceScale: 3,
        },
      },
      {
        id: 'SG161222/RealVisXL_V5.0',
        label: 'RealVisXL V5.0',
        operation: 'text-to-image',
        notes: 'Photorealistic SDXL finetune for portrait and realistic scenes.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['photoreal', 'sdxl'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'float16',
          lowVram: true,
          steps: 35,
          guidanceScale: 6,
          negativePrompt:
            'bad hands, bad anatomy, ugly, deformed, face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth',
        },
      },
      {
        id: 'RunDiffusion/Juggernaut-XL-v9',
        label: 'Juggernaut XL v9',
        operation: 'text-to-image',
        notes: 'Photoreal/cinematic SDXL finetune for people, interiors, food, and landscapes.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['photoreal', 'cinematic', 'sdxl'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'float16',
          lowVram: true,
          steps: 30,
          guidanceScale: 6,
        },
      },
      {
        id: 'Vargol/ProteusV0.2',
        label: 'Proteus V0.2 fp16',
        operation: 'text-to-image',
        notes:
          'FP16 SDXL-style finetune for detailed creative images; GPL-3.0 licensed conversion.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['sdxl', 'creative'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'float16',
          lowVram: true,
          steps: 35,
          guidanceScale: 7,
        },
      },
      {
        id: 'qwen-image-lightning-8step',
        modelId: 'Qwen/Qwen-Image',
        label: 'Qwen Image Lightning 8-step',
        operation: 'text-to-image',
        notes: 'Qwen Image with Lightning LoRA for much faster local generation.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['qwen', 'fast', 'lora', 'workflow'],
        workflow: {
          id: 'qwen-image-lightning',
          label: 'Qwen Image + Lightning LoRA',
          description:
            'Loads Qwen Image, applies Qwen Image Lightning LoRA, and uses FlowMatch Euler.',
        },
        params: {
          pipeline: 'qwen-image',
          dtype: 'bfloat16',
          lowVram: true,
          loraPath: 'lightx2v/Qwen-Image-Lightning',
          loraWeightName: 'Qwen-Image-Lightning-8steps-V1.0.safetensors',
          scheduler: 'qwen-lightning-flowmatch-euler',
          steps: 8,
          guidanceScale: 1,
          trueCfgScale: 1,
        },
      },
      {
        id: 'Qwen/Qwen-Image',
        label: 'Qwen Image',
        operation: 'text-to-image',
        notes: 'High quality Qwen text-to-image model; best on 24GB+ CUDA GPUs.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['qwen', 'high-quality'],
        workflow: {
          id: 'qwen-image',
          label: 'Qwen Image',
          description: 'Loads Qwen Image through DiffusionPipeline with bfloat16 weights.',
        },
        params: {
          pipeline: 'qwen-image',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 50,
          guidanceScale: 4,
          trueCfgScale: 4,
        },
      },
      {
        id: 'Efficient-Large-Model/Sana_600M_1024px_diffusers',
        label: 'Sana 600M 1024px',
        operation: 'text-to-image',
        notes: 'Lightweight 1024px text-to-image model for 8GB class GPUs.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda'],
        tags: ['lightweight', '1024px'],
        workflow: {
          id: 'sana-diffusers',
          label: 'Sana Pipeline',
          description:
            'Loads Sana through SanaPipeline when the installed diffusers build supports it.',
        },
        params: {
          pipeline: 'sana',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 20,
          guidanceScale: 4.5,
        },
      },
      {
        id: 'Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers',
        label: 'Sana 1.6B 1024px BF16',
        operation: 'text-to-image',
        notes: 'Higher quality Sana preset for 12GB+ CUDA GPUs.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['high-quality', '1024px'],
        workflow: {
          id: 'sana-diffusers',
          label: 'Sana Pipeline',
          description:
            'Loads Sana through SanaPipeline when the installed diffusers build supports it.',
        },
        params: {
          pipeline: 'sana',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 20,
          guidanceScale: 4.5,
        },
      },
      {
        id: 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        label: 'PixArt-Sigma XL 1024',
        operation: 'text-to-image',
        notes: 'Transformer-based 1024px model with strong prompt following.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['dit', '1024px', 'high-quality'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'float16',
          lowVram: true,
          steps: 20,
          guidanceScale: 4.5,
        },
      },
      {
        id: 'Kwai-Kolors/Kolors-diffusers',
        label: 'Kolors',
        operation: 'text-to-image',
        notes:
          'Photorealistic English/Chinese prompt model; commercial use has extra registration terms.',
        minVramMiB: 16384,
        recommendedVramMiB: 24576,
        deviceKinds: ['cuda'],
        tags: ['photoreal', 'chinese', 'high-quality'],
        workflow: {
          id: 'kolors-t2i',
          label: 'Kolors Pipeline',
          description:
            'Loads Kolors through KolorsPipeline when the installed diffusers build supports it.',
        },
        params: {
          pipeline: 'kolors',
          variant: 'fp16',
          dtype: 'float16',
          lowVram: true,
          steps: 50,
          guidanceScale: 5,
        },
      },
      {
        id: 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled',
        label: 'HunyuanDiT v1.2 Distilled',
        operation: 'text-to-image',
        notes: '25-step distilled DiT model with English/Chinese prompt support.',
        minVramMiB: 16384,
        recommendedVramMiB: 24576,
        deviceKinds: ['cuda'],
        tags: ['dit', 'chinese', 'distilled'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'float16',
          lowVram: true,
          steps: 25,
          guidanceScale: 7.5,
        },
      },
      {
        id: 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
        label: 'HunyuanDiT v1.2',
        operation: 'text-to-image',
        notes: 'Full Hunyuan DiT model for higher quality English/Chinese generation.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['dit', 'chinese', 'high-quality'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'float16',
          lowVram: true,
          steps: 50,
          guidanceScale: 7.5,
        },
      },
      {
        id: 'stabilityai/stable-diffusion-3.5-medium',
        label: 'Stable Diffusion 3.5 Medium',
        operation: 'text-to-image',
        notes: 'Quality/resource-balanced SD 3.5 preset; may require Hugging Face access approval.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['high-quality', 'sd3.5'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 40,
          guidanceScale: 4.5,
        },
      },
      {
        id: 'stabilityai/stable-diffusion-3.5-large-turbo',
        label: 'Stable Diffusion 3.5 Large Turbo',
        operation: 'text-to-image',
        notes: 'High quality low-step SD 3.5 preset; may require Hugging Face access approval.',
        minVramMiB: 16384,
        recommendedVramMiB: 24576,
        deviceKinds: ['cuda'],
        tags: ['fast', 'high-quality', 'sd3.5'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 4,
          guidanceScale: 0,
        },
      },
      {
        id: 'stabilityai/stable-diffusion-3.5-large',
        label: 'Stable Diffusion 3.5 Large',
        operation: 'text-to-image',
        notes: 'High quality SD 3.5 model; requires Hugging Face access approval and high VRAM.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['high-quality', 'sd3.5', 'large'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 28,
          guidanceScale: 3.5,
          maxSequenceLength: 512,
        },
      },
      {
        id: 'black-forest-labs/FLUX.1-schnell',
        label: 'FLUX.1 schnell',
        operation: 'text-to-image',
        notes: 'Large text-to-image model; best on 16GB+ GPUs.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['flux', 'large'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: { pipeline: 'diffusion-pipeline', dtype: 'bfloat16', steps: 4, guidanceScale: 0 },
      },
      {
        id: 'black-forest-labs/FLUX.1-dev',
        label: 'FLUX.1 dev',
        operation: 'text-to-image',
        notes:
          'High quality FLUX preset; best on 24GB+ CUDA GPUs and subject to model license/access.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['flux', 'high-quality', 'large'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 28,
          guidanceScale: 3.5,
        },
      },
      {
        id: 'kandinsky-community/kandinsky-3',
        label: 'Kandinsky 3',
        operation: 'text-to-image',
        notes: 'Multilingual high quality model with a large text encoder; needs high VRAM.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['multilingual', 'high-quality', 'large'],
        workflow: {
          id: 'diffusers-generic-t2i',
          label: 'Diffusers Pipeline',
          description: 'Loads the model through the generic DiffusionPipeline path.',
        },
        params: {
          pipeline: 'diffusion-pipeline',
          variant: 'fp16',
          dtype: 'float16',
          lowVram: true,
          steps: 50,
          guidanceScale: 4,
        },
      },
      {
        id: DEFAULT_MODELS['image-to-image'],
        label: 'InstructPix2Pix edit',
        operation: 'image-to-image',
        notes:
          'Instruction-following image edit preset for prompts such as making a subject smile or turning an image grayscale.',
        minVramMiB: 4096,
        recommendedVramMiB: 6144,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['edit', 'instruction', 'sd15'],
        workflow: {
          id: 'instruct-pix2pix',
          label: 'InstructPix2Pix',
          description:
            'Uses the selected image as an edit source and applies the prompt as an instruction.',
        },
        params: {
          pipeline: 'instruct-pix2pix',
          dtype: 'float16',
          steps: 30,
          guidanceScale: 7.5,
          imageGuidanceScale: 1.5,
          strength: 0.8,
        },
      },
      {
        id: 'stabilityai/sd-turbo',
        label: 'SD Turbo img2img',
        operation: 'image-to-image',
        notes: 'Fast image variation pipeline for low VRAM GPUs.',
        minVramMiB: 4096,
        recommendedVramMiB: 6144,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['fast', 'sd15', 'variation'],
        workflow: {
          id: 'diffusers-auto-i2i',
          label: 'Diffusers img2img',
          description: 'Uses the selected image as a variation source.',
        },
        params: { steps: 2, guidanceScale: 0, strength: 0.65 },
      },
      {
        id: 'stabilityai/sdxl-turbo',
        label: 'SDXL Turbo img2img',
        operation: 'image-to-image',
        notes: 'Higher quality image editing preset for 8GB+ GPUs.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['fast', 'sdxl'],
        workflow: {
          id: 'diffusers-auto-i2i',
          label: 'Diffusers img2img',
          description: 'Uses the selected image as the image-to-image source.',
        },
        params: { steps: 4, guidanceScale: 0, strength: 0.65 },
      },
      {
        id: 'Lykon/DreamShaper',
        label: 'DreamShaper img2img',
        operation: 'image-to-image',
        notes: 'Lighter creative img2img preset for low VRAM edits.',
        minVramMiB: 4096,
        recommendedVramMiB: 6144,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['lightweight', 'sd15', 'edit'],
        workflow: {
          id: 'diffusers-auto-i2i',
          label: 'Diffusers img2img',
          description: 'Uses the selected image as the image-to-image source.',
        },
        params: { dtype: 'float16', steps: 25, guidanceScale: 7, strength: 0.65 },
      },
      {
        id: 'stabilityai/stable-diffusion-xl-base-1.0',
        label: 'SDXL Base img2img',
        operation: 'image-to-image',
        notes: 'General purpose SDXL image variation/edit preset.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['sdxl', 'edit'],
        workflow: {
          id: 'diffusers-auto-i2i',
          label: 'Diffusers img2img',
          description: 'Uses the selected image as the image-to-image source.',
        },
        params: {
          variant: 'fp16',
          dtype: 'float16',
          useSafetensors: true,
          steps: 30,
          guidanceScale: 7,
          strength: 0.65,
        },
      },
      {
        id: 'stabilityai/stable-diffusion-xl-refiner-1.0',
        label: 'SDXL Refiner img2img',
        operation: 'image-to-image',
        notes: 'Refines an existing Canvas image with an SDXL image-to-image pipeline.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['sdxl', 'refine', 'edit'],
        workflow: {
          id: 'diffusers-auto-i2i',
          label: 'Diffusers img2img',
          description: 'Uses the selected image as the image-to-image source.',
        },
        params: {
          variant: 'fp16',
          dtype: 'float16',
          useSafetensors: true,
          steps: 30,
          guidanceScale: 7,
          strength: 0.35,
        },
      },
      {
        id: 'Kwai-Kolors/Kolors-diffusers',
        label: 'Kolors img2img',
        operation: 'image-to-image',
        notes: 'Kolors image-to-image edit path for English/Chinese prompts on high VRAM GPUs.',
        minVramMiB: 16384,
        recommendedVramMiB: 24576,
        deviceKinds: ['cuda'],
        tags: ['photoreal', 'chinese', 'edit'],
        workflow: {
          id: 'kolors-img2img',
          label: 'Kolors img2img',
          description: 'Loads Kolors through KolorsImg2ImgPipeline and passes the selected image.',
        },
        params: {
          pipeline: 'kolors-img2img',
          variant: 'fp16',
          dtype: 'float16',
          lowVram: true,
          steps: 30,
          guidanceScale: 5,
          strength: 0.65,
        },
      },
      {
        id: 'Qwen/Qwen-Image-Edit',
        label: 'Qwen Image Edit',
        operation: 'image-to-image',
        notes:
          'Prompt-guided image editing workflow; requires a newer diffusers build and high VRAM.',
        minVramMiB: 16384,
        recommendedVramMiB: 24576,
        deviceKinds: ['cuda'],
        tags: ['qwen', 'edit', 'workflow'],
        workflow: {
          id: 'qwen-image-edit',
          label: 'Qwen Image Edit',
          description:
            'Loads Qwen Image Edit through DiffusionPipeline and passes the selected image.',
        },
        params: {
          pipeline: 'qwen-image-edit',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 20,
          trueCfgScale: 4,
        },
      },
      {
        id: 'Qwen/Qwen-Image-Edit-2509',
        label: 'Qwen Image Edit 2509',
        operation: 'image-to-image',
        notes: 'Newer Qwen image editing workflow; accepts the selected Canvas image and prompt.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['qwen', 'edit', 'workflow', 'high-quality'],
        workflow: {
          id: 'qwen-image-edit-plus',
          label: 'Qwen Image Edit Plus',
          description:
            'Loads the newer Qwen edit pipeline and passes the selected Canvas image as an edit input.',
        },
        params: {
          pipeline: 'qwen-image-edit-plus',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 50,
          guidanceScale: 4,
          trueCfgScale: 4,
          strength: 0.8,
        },
      },
      {
        id: DEFAULT_MODELS.inpaint,
        label: 'Stable Diffusion Inpainting',
        operation: 'inpaint',
        notes: 'Mask-based image editing with a lighter SD 1.5 inpaint model.',
        minVramMiB: 6144,
        recommendedVramMiB: 8192,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['inpaint', 'sd15'],
        workflow: {
          id: 'diffusers-inpaint',
          label: 'Diffusers Inpaint',
          description: 'Uses an input image plus mask image.',
        },
        params: { dtype: 'float16', steps: 30, guidanceScale: 7.5, strength: 0.8 },
      },
      {
        id: 'stabilityai/stable-diffusion-2-inpainting',
        label: 'Stable Diffusion 2 Inpaint',
        operation: 'inpaint',
        notes: 'Moderate VRAM inpainting model based on Stable Diffusion 2.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda', 'mps'],
        tags: ['inpaint', 'sd2'],
        workflow: {
          id: 'diffusers-inpaint',
          label: 'Diffusers Inpaint',
          description: 'Uses an input image plus mask image.',
        },
        params: { dtype: 'float16', steps: 30, guidanceScale: 7.5, strength: 0.8 },
      },
      {
        id: 'Lykon/dreamshaper-7-inpainting',
        label: 'DreamShaper Inpaint',
        operation: 'inpaint',
        notes: 'Creative low VRAM inpainting model fine-tuned from the SD inpaint family.',
        minVramMiB: 6144,
        recommendedVramMiB: 8192,
        deviceKinds: ['cuda', 'mps', 'cpu'],
        tags: ['inpaint', 'sd15', 'art'],
        workflow: {
          id: 'diffusers-inpaint',
          label: 'Diffusers Inpaint',
          description: 'Uses an input image plus mask image.',
        },
        params: { dtype: 'float16', steps: 25, guidanceScale: 7, strength: 0.8 },
      },
      {
        id: 'qwen-image-edit-inpaint',
        modelId: 'Qwen/Qwen-Image-Edit',
        label: 'Qwen Image Edit Inpaint',
        operation: 'inpaint',
        notes: 'Prompt-guided Qwen inpainting workflow; requires a recent diffusers build.',
        minVramMiB: 24576,
        recommendedVramMiB: 49152,
        deviceKinds: ['cuda'],
        tags: ['qwen', 'inpaint', 'workflow'],
        workflow: {
          id: 'qwen-image-edit-inpaint',
          label: 'Qwen Image Edit Inpaint',
          description: 'Uses the input image plus mask through QwenImageEditInpaintPipeline.',
        },
        params: {
          pipeline: 'qwen-image-edit-inpaint',
          dtype: 'bfloat16',
          lowVram: true,
          steps: 50,
          guidanceScale: 4,
          trueCfgScale: 4,
          strength: 0.8,
        },
      },
      {
        id: 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        label: 'SDXL Inpainting',
        operation: 'inpaint',
        notes: 'Higher quality mask editing; works best on 16GB class GPUs.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda', 'mps'],
        tags: ['inpaint', 'sdxl'],
        workflow: {
          id: 'diffusers-inpaint',
          label: 'Diffusers Inpaint',
          description: 'Uses an input image plus mask image.',
        },
        params: { dtype: 'float16', steps: 30, guidanceScale: 7.5, strength: 0.8 },
      },
      {
        id: DEFAULT_MODELS['text-to-video'],
        label: 'ZeroScope v2 576w',
        operation: 'text-to-video',
        notes: 'Text-to-video baseline that fits 30 frames around 8GB VRAM.',
        minVramMiB: 8192,
        recommendedVramMiB: 12288,
        deviceKinds: ['cuda'],
        tags: ['video'],
        workflow: {
          id: 'text-to-video-sd',
          label: 'TextToVideoSD',
          description: 'Generates video frames from a text prompt.',
        },
        params: {
          pipeline: 'text-to-video-sd',
          dtype: 'float16',
          width: 576,
          height: 320,
          frames: 24,
          fps: 8,
          steps: 25,
          guidanceScale: 7.5,
        },
      },
      {
        id: DEFAULT_MODELS['image-to-video'],
        label: 'Stable Video Diffusion XT',
        operation: 'image-to-video',
        notes: 'Image-conditioned short video generation.',
        minVramMiB: 12288,
        recommendedVramMiB: 16384,
        deviceKinds: ['cuda'],
        tags: ['video'],
        workflow: {
          id: 'stable-video-diffusion',
          label: 'Stable Video Diffusion',
          description: 'Animates the selected Canvas image with SVD image-to-video.',
        },
        params: {
          pipeline: 'stable-video-diffusion',
          variant: 'fp16',
          dtype: 'float16',
          width: 1024,
          height: 576,
          frames: 25,
          fps: 7,
          steps: 25,
          guidanceScale: 1.5,
          motionBucketId: 180,
          noiseAugStrength: 0.1,
          decodeChunkSize: 8,
        },
      },
    ],
  },
  async run(input) {
    const params = parseParams(input.params)
    const venvDir = defaultVenvDir(params)
    const workDir = defaultWorkDir(params)
    const outputDir = join(workDir, input.jobId)
    const python = await ensureVenv(params, venvDir, input.signal)
    const script = await writeGeneratorScript(workDir)
    const inputPath = join(workDir, `${input.jobId}.json`)
    await writeFile(
      inputPath,
      JSON.stringify({
        jobId: input.jobId,
        prompt: input.prompt,
        operation: params.operation,
        model: params.model,
        variant: params.variant,
        outputDir,
        width: params.width,
        height: params.height,
        count: params.count ?? params.batchSize,
        seed: params.seed,
        negativePrompt: params.negativePrompt,
        imagePath: params.imagePath,
        imageDataUrl: params.imageDataUrl,
        maskPath: params.maskPath,
        maskDataUrl: params.maskDataUrl,
        strength: params.strength,
        numFrames: params.numFrames ?? params.frames,
        fps: params.fps,
        outputFormat: params.outputFormat,
        lowVram: params.lowVram !== false,
        disableSafetyChecker: params.disableSafetyChecker === true,
        allowCpu: params.allowCpu === true,
        device: params.device,
        gpuIndex: params.gpuIndex,
        pipeline: params.pipeline,
        dtype: params.dtype,
        loraPath: params.loraPath,
        loraWeightName: params.loraWeightName,
        loraScale: params.loraScale,
        scheduler: params.scheduler,
        schedulerTimestepSpacing: params.schedulerTimestepSpacing,
        trueCfgScale: params.trueCfgScale,
        imageGuidanceScale: params.imageGuidanceScale,
        motionBucketId: params.motionBucketId,
        noiseAugStrength: params.noiseAugStrength,
        decodeChunkSize: params.decodeChunkSize,
        steps: params.steps,
        guidanceScale: params.guidanceScale,
        maxSequenceLength: params.maxSequenceLength,
        useSafetensors: params.useSafetensors,
      }),
      'utf-8',
    )

    input.onProgress(0.05)
    const result = await runProcess(python, [script], {
      timeoutMs: params.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      signal: input.signal,
      env: {
        ...process.env,
        ...(params.device === 'cuda' && params.gpuIndex !== undefined
          ? { CUDA_VISIBLE_DEVICES: String(params.gpuIndex) }
          : {}),
        SEPILOTD_IMAGE_GEN_INPUT: inputPath,
      },
    })
    const metadata = parseMetadata(result.stdout)
    input.onProgress(0.95)

    const outputs = []
    for (let index = 0; index < metadata.outputs.length; index += 1) {
      const output = metadata.outputs[index]!
      const bytes = await readFile(output.path)
      if (bytes.length === 0) {
        throw new Error(`generated image is empty: ${output.path}`)
      }
      const mime = output.mime ?? 'image/png'
      const kind = output.kind ?? (mime.startsWith('video/') ? 'video' : 'image')
      outputs.push({
        id: kind === 'video' ? `${input.jobId}-video-${index}` : `${input.jobId}-${index}`,
        mime,
        bytes,
        path: output.path,
        kind,
      })
    }
    input.onProgress(1)
    return { outputs }
  },
}
