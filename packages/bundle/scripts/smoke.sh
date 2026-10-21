#!/usr/bin/env bash
set -euo pipefail
BIN="${1:?usage: smoke.sh <path-to-binary>}"
exec node --input-type=module - "$BIN" "${SEPILOT_SMOKE_PORT:-0}" <<'NODE'
import { createHash } from 'node:crypto'
import { execFile, spawn } from 'node:child_process'
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'

const runFile = promisify(execFile)
const controller = new AbortController()
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => controller.abort(new Error(`Smoke interrupted by ${signal}`)))
}
let profile
let child
let childError

function exited() {
  return child?.exitCode !== null || child?.signalCode !== null
}

async function waitForExit(timeoutMs) {
  if (!child || exited()) return
  await new Promise(resolve => {
    const onExit = () => { clearTimeout(timer); resolve() }
    const timer = setTimeout(() => { child.removeListener('exit', onExit); resolve() }, timeoutMs)
    child.once('exit', onExit)
  })
}

async function cleanup() {
  if (child && !exited()) {
    child.kill('SIGTERM')
    await waitForExit(5000)
    if (!exited()) {
      child.kill('SIGKILL')
      await waitForExit(3000)
    }
    if (!exited()) throw new Error('Owned smoke daemon did not exit')
  }
  if (profile) await rm(profile, { recursive: true, force: true })
}

async function selectPort(requested) {
  if (!/^\d+$/.test(requested) || Number(requested) > 65535) throw new Error('Invalid smoke port')
  const server = createServer()
  await new Promise((resolve, reject) => {
    server.once('error', () => reject(new Error('Requested smoke port is unavailable')))
    server.listen(Number(requested), '127.0.0.1', resolve)
  })
  const port = server.address().port
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  return port
}

async function main() {
  const artifact = await realpath(process.argv[2])
  const port = await selectPort(process.argv[3])
  controller.signal.throwIfAborted()
  const digest = createHash('sha256').update(await readFile(artifact)).digest('hex')
  profile = await mkdtemp(join(tmpdir(), 'sepilot-native-smoke-'))
  await mkdir(join(profile, 'home'))
  await mkdir(join(profile, 'workspace'))
  await mkdir(join(profile, 'security'))
  await writeFile(join(profile, 'security', 'daemon.token'), 'smoke-test-token\n', { mode: 0o600 })
  const env = {
    PATH: process.env.PATH,
    HOME: join(profile, 'home'),
    SEPILOTD_HOME: profile,
    SEPILOTD_DATA_DIR: profile,
    SEPILOTD_HOST: '127.0.0.1',
    SEPILOTD_PORT: String(port),
    SEPILOTD_URL: `http://127.0.0.1:${port}`,
    SEPILOT_NO_AUTO_START: '1',
  }
  const runArtifact = async command => {
    controller.signal.throwIfAborted()
    const result = await runFile(artifact, [command], {
      cwd: join(profile, 'workspace'), env, signal: controller.signal, timeout: 15000,
    })
    process.stdout.write(result.stdout)
  }
  console.log('== version ==')
  await runArtifact('--version')
  console.log('== owned daemon startup ==')
  child = spawn(artifact, ['__daemon'], { cwd: join(profile, 'workspace'), env, stdio: ['ignore', 'pipe', 'pipe'] })
  child.once('error', error => { childError = error })
  let pendingLog = ''
  let listening = false
  child.stdout.setEncoding('utf8')
  child.stdout.on('data', chunk => {
    pendingLog += chunk
    const lines = pendingLog.split('\n')
    pendingLog = lines.pop().slice(-65536)
    for (const line of lines) {
      try {
        const text = line.match(/^\S+ \[INFO\s*\] \[main\] Server started (.+)$/)
        const entry = JSON.parse(text ? text[1] : line)
        if ((text || (entry.component === 'main' && entry.msg === 'Server started'))
          && entry.host === '127.0.0.1' && entry.port === port) listening = true
      } catch {}
    }
  })
  child.stderr.resume()
  const verifyChild = async () => {
    if (childError) throw childError
    if (exited()) throw new Error('Owned smoke daemon exited before verification')
    const recorded = (await readFile(join(profile, 'sepilotd.pid'), 'utf8')).trim()
    if (recorded !== String(child.pid)) throw new Error('Smoke profile PID does not match the launched artifact')
  }
  // Loaded hosts can take longer than the default 30 s to start the owned daemon;
  // SEPILOT_SMOKE_START_MS only extends the wait, every check still runs.
  const startBudget = Math.max(30000, Number.parseInt(process.env.SEPILOT_SMOKE_START_MS ?? '', 10) || 0)
  const deadline = Date.now() + startBudget
  while (!listening) {
    controller.signal.throwIfAborted()
    if (childError) throw childError
    if (exited()) throw new Error('Owned smoke daemon exited before listening')
    if (Date.now() >= deadline) throw new Error('Owned smoke daemon did not report successful listening')
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  await verifyChild()
  console.log('== health ==')
  const response = await fetch(`${env.SEPILOTD_URL}/api/v1/health`, {
    headers: { authorization: 'Bearer smoke-test-token' },
    signal: AbortSignal.any([controller.signal, AbortSignal.timeout(2000)]),
  })
  if (!response.ok) throw new Error('Owned smoke daemon health failed')
  await response.arrayBuffer()
  await verifyChild()
  console.log('health OK')
  console.log('== status ==')
  await runArtifact('status')
  await verifyChild()
  const finalDigest = createHash('sha256').update(await readFile(artifact)).digest('hex')
  if (finalDigest !== digest) throw new Error('Smoke artifact changed during verification')
  console.log(`artifact SHA-256 ${digest}`)
}

try {
  await main()
} catch (error) {
  console.error(error.message)
  process.exitCode = 1
} finally {
  try { await cleanup() } catch (error) { console.error(error.message); process.exitCode = 1 }
}
if (!process.exitCode) console.log('smoke OK')
NODE
