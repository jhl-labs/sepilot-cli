#!/usr/bin/env bun
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { compileWithBun } from './bun-compile.ts'

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url))
const EXPECTED_OUTPUT = 'BUN_NETWORK_DISPATCHER_OK'

export function currentBunCompileTarget(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string {
  const platformName = platform === 'win32' ? 'windows' : platform
  if (!['windows', 'linux', 'darwin'].includes(platformName)) {
    throw new Error(`Unsupported Bun dispatcher smoke platform: ${platform}`)
  }
  if (!['x64', 'arm64'].includes(arch)) {
    throw new Error(`Unsupported Bun dispatcher smoke architecture: ${arch}`)
  }
  return `bun-${platformName}-${arch}`
}

export async function runBunNetworkDispatcherSmoke(): Promise<void> {
  const tempDir = mkdtempSync(join(tmpdir(), 'sepilot-bun-network-smoke-'))
  const executable = join(
    tempDir,
    process.platform === 'win32' ? 'dispatcher-smoke.exe' : 'dispatcher-smoke',
  )
  const entry = resolve(
    SCRIPT_DIR,
    'bun-network-dispatcher-smoke-entry.ts',
  )

  try {
    await compileWithBun(currentBunCompileTarget(), entry, executable, Bun)
    const output = execFileSync(executable, [], {
      encoding: 'utf8',
      timeout: 15_000,
      windowsHide: true,
    })
    if (!output.includes(EXPECTED_OUTPUT)) {
      throw new Error(`Bun dispatcher smoke returned unexpected output: ${output.trim()}`)
    }
    process.stdout.write(`${EXPECTED_OUTPUT}\n`)
  } finally {
    rmSync(tempDir, { recursive: true, force: true })
  }
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : undefined
if (invokedPath === import.meta.url) {
  await runBunNetworkDispatcherSmoke()
}
