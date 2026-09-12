import { execFile } from 'node:child_process'
import os from 'node:os'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

export interface ImageGenHardwareDevice {
  id: string
  label: string
  kind: 'cuda' | 'mps' | 'cpu'
  index?: number
  name: string
  memoryTotalMiB: number | null
  memoryFreeMiB: number | null
  available: boolean
}

export interface ImageGenHardwareSnapshot {
  collectedAt: string
  devices: ImageGenHardwareDevice[]
  gpuAvailable: boolean
  reason?: string
}

function parseGpuNumber(value: string | undefined): number | null {
  if (!value) return null
  const normalized = value.trim()
  if (!normalized || normalized === '[N/A]' || normalized === 'N/A') return null
  const parsed = Number(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

async function collectCudaDevices(timeoutMs = 3_000): Promise<{
  devices: ImageGenHardwareDevice[]
  reason?: string
}> {
  try {
    const { stdout } = await execFileAsync(
      'nvidia-smi',
      ['--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
      {
        timeout: timeoutMs,
        maxBuffer: 1024 * 1024,
      },
    )

    const devices = stdout
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line, fallbackIndex) => {
        const parts = line.split(',').map((part) => part.trim())
        const index = parseGpuNumber(parts[0]) ?? fallbackIndex
        const name = parts[1] || `CUDA GPU ${index}`
        const total = parseGpuNumber(parts[2])
        const free = parseGpuNumber(parts[3])
        const totalLabel = total ? `${Math.round(total / 1024)}GB` : 'VRAM unknown'
        return {
          id: `cuda:${index}`,
          label: `CUDA ${index} · ${name} · ${totalLabel}`,
          kind: 'cuda' as const,
          index,
          name,
          memoryTotalMiB: total,
          memoryFreeMiB: free,
          available: true,
        } satisfies ImageGenHardwareDevice
      })

    return {
      devices,
      reason: devices.length > 0 ? undefined : 'nvidia-smi returned no GPU rows',
    }
  } catch (error) {
    return {
      devices: [],
      reason: error instanceof Error ? error.message : String(error),
    }
  }
}

export async function collectImageGenHardware(): Promise<ImageGenHardwareSnapshot> {
  const cuda = await collectCudaDevices()
  const devices: ImageGenHardwareDevice[] = [...cuda.devices]

  if (process.platform === 'darwin') {
    devices.push({
      id: 'mps',
      label: 'Apple GPU (MPS)',
      kind: 'mps',
      name: 'Apple GPU',
      memoryTotalMiB: Math.round(os.totalmem() / 1024 / 1024),
      memoryFreeMiB: Math.round(os.freemem() / 1024 / 1024),
      available: true,
    })
  }

  devices.push({
    id: 'cpu',
    label: 'CPU',
    kind: 'cpu',
    name: os.cpus()[0]?.model ?? 'CPU',
    memoryTotalMiB: Math.round(os.totalmem() / 1024 / 1024),
    memoryFreeMiB: Math.round(os.freemem() / 1024 / 1024),
    available: true,
  })

  return {
    collectedAt: new Date().toISOString(),
    devices,
    gpuAvailable: devices.some((device) => device.kind === 'cuda' || device.kind === 'mps'),
    reason: cuda.devices.length > 0 ? undefined : cuda.reason,
  }
}
