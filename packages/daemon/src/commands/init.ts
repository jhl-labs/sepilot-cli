import { mkdir, writeFile, access } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { randomUUID, randomBytes, generateKeyPairSync } from 'node:crypto'
import YAML from 'yaml'
import { createDefaultPolicy } from '../security/policy-engine.js'

export async function initCommand(options?: {
  dataDir?: string
  deviceName?: string
  role?: string
}): Promise<void> {
  const dataDir = options?.dataDir ?? process.env.SEPILOTD_DATA_DIR ?? join(homedir(), '.sepilotd')
  const role = options?.role ?? 'desktop'
  if (!['desktop', 'server', 'edge'].includes(role)) {
    throw new Error(`Invalid device role '${role}'. Expected one of: desktop, server, edge.`)
  }

  // Check if already initialized
  try {
    await access(join(dataDir, 'config.yaml'))
    console.log(`sepilotd is already initialized at ${dataDir}`)
    console.log('To reinitialize, delete the directory first.')
    return
  } catch {
    // Not initialized yet — proceed
  }

  console.log(`Initializing sepilotd at ${dataDir}...\n`)

  // Create directory structure
  const dirs = ['sessions', 'memory', 'skills', 'security', 'cache', 'logs']
  for (const dir of dirs) {
    await mkdir(join(dataDir, dir), { recursive: true })
  }

  // Generate device ID
  const deviceId = randomUUID()
  const deviceName = options?.deviceName ?? homedir().split('/').pop() ?? 'default'
  // Generate Ed25519 key pair
  const { publicKey, privateKey } = generateKeyPairSync('ed25519', {
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
  })
  await writeFile(join(dataDir, 'security', 'device.key'), privateKey, { mode: 0o600 })
  await writeFile(join(dataDir, 'security', 'device.pub'), publicKey, { mode: 0o644 })

  // Generate data encryption key (32 random bytes)
  const dataKey = randomBytes(32)
  await writeFile(join(dataDir, 'security', 'data.key'), dataKey, { mode: 0o600 })

  // Generate auth tokens
  const daemonToken = randomBytes(32).toString('hex')
  const gatewayToken = randomBytes(32).toString('hex')
  await writeFile(join(dataDir, 'security', 'daemon.token'), daemonToken, { mode: 0o600 })
  await writeFile(join(dataDir, 'security', 'gateway.token'), gatewayToken, { mode: 0o600 })

  // Create default config
  const config = {
    version: 1,
    device: { id: deviceId, name: deviceName, role },
    daemon: { port: 17600, host: '127.0.0.1', resumeArtifactRetentionDays: 30 },
    gateway: { url: 'http://127.0.0.1:17610' },
    providers: [
      {
        id: 'ollama',
        type: 'ollama',
        baseUrl: 'http://localhost:11434',
        models: ['llama3.3'],
        default: true,
      },
    ],
    agent: {
      autonomy: 'supervised',
      thinkingLevel: 'medium',
      capabilities: { hostSystemInfo: true },
    },
    channels: [],
    memory: { encryption: true, vectorBackend: 'auto' },
    security: { toolPolicy: 'policies.yaml', auditLog: true, sandbox: 'local' },
    observability: { telemetry: false },
  }
  await writeFile(join(dataDir, 'config.yaml'), YAML.stringify(config), { mode: 0o600 })

  // Create default policies
  const policies = createDefaultPolicy()
  await writeFile(join(dataDir, 'security', 'policies.yaml'), YAML.stringify(policies), {
    mode: 0o644,
  })

  // Auto-detect Ollama
  console.log('\nChecking for local LLM...')
  try {
    const ollamaRes = await fetch('http://localhost:11434/api/tags')
    if (ollamaRes.ok) {
      const data = (await ollamaRes.json()) as { models?: Array<{ name: string }> }
      const models = data.models?.map((m) => m.name) ?? []
      if (models.length > 0) {
        const selectedModels = models.slice(0, 5)
        console.log(`  Ollama detected: ${models.length} models`)
        console.log(`  Selected: ${selectedModels.join(', ')}`)
        config.providers[0].models = selectedModels
        await writeFile(join(dataDir, 'config.yaml'), YAML.stringify(config), { mode: 0o600 })
      } else {
        console.log('  Ollama running but no models. Run: ollama pull llama3.3')
      }
    }
  } catch {
    console.log('  Ollama not detected. Install from https://ollama.com')
  }

  // Create sample skills
  console.log('\nCreating sample skills...')
  const sampleSkills = [
    {
      name: 'git-status',
      content: `+++
name = "git-status"
version = "1.0.0"
description = "Check git repository status and recent commits"
tools = ["terminal.run"]
tags = ["git", "development"]
+++

# Git Status

Check the current git repository status.

## Steps

1. Run \`git status\` to see working tree status
2. Run \`git log --oneline -10\` to see recent commits
3. Summarize the state of the repository
`,
    },
    {
      name: 'system-info',
      content: `+++
name = "system-info"
version = "1.0.0"
description = "Gather system information (OS, CPU, memory, disk)"
tools = ["terminal.run"]
tags = ["system", "monitoring"]
+++

# System Info

Gather and report system information with \`system.info\` when the
host-system-info capability exposes that tool. It collects CPU, memory,
storage, uptime, and optional GPU details in a single deterministic call. Fall
back to \`terminal.run\` only when \`system.info\` is unavailable.
Do not use this skill for requests to implement, write, build, or debug a
program, script, dashboard, or monitoring tool. For those requests, implement
the requested code instead of collecting this host's current metrics.
Do not send interim progress updates like "I'll check memory next." Complete the
required checks first, then return one final report. If a step is blocked, state
that clearly in the final answer.

## Steps

1. Prefer \`system.info\` with \`includeGpu: true\` when the user asked for GPU too.
2. If \`system.info\` is unavailable, fall back to direct \`terminal.run\` calls:
   \`uname -a\`, \`nproc\`, \`lscpu\`, \`free -h\`, \`df -h\`, and \`uptime\`.
3. Summarize findings in a clean report.
`,
    },
  ]

  for (const skill of sampleSkills) {
    const skillDir = join(dataDir, 'skills', skill.name)
    await mkdir(skillDir, { recursive: true })
    await writeFile(join(skillDir, 'SKILL.md'), skill.content)
    console.log(`  Created skill: ${skill.name}`)
  }

  // Print summary
  console.log('Created directories:')
  for (const dir of dirs) console.log(`  ${dataDir}/${dir}/`)
  console.log('')
  console.log('Generated:')
  console.log(`  Device ID:    ${deviceId}`)
  console.log(`  Device name:  ${deviceName}`)
  console.log(`  Role:         ${role}`)
  console.log(`  Device keys:  security/device.key, security/device.pub`)
  console.log(`  Data key:     security/data.key`)
  console.log(`  Auth tokens:  security/daemon.token, security/gateway.token`)
  console.log(`  Config:       config.yaml`)
  console.log(`  Policies:     security/policies.yaml`)
  console.log('')
  console.log('Next steps:')
  console.log('  1. Edit config.yaml to add LLM providers (API keys, models)')
  console.log('  2. Start the daemon: sepilotd')
  console.log('  3. Connect via CLI: sepilot status')
  console.log('')
  console.log('sepilotd initialized successfully!')
}
