import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { GatewayClient } from '@sepilotd/api-client'
import { resolveGatewayBaseUrl } from '../client/token.js'
import { formatHealthComponentIcon } from './health-display.js'
import { getOutputFormat } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const TEST_CONNECTION_COPY = {
  en: {
    testingConnections: 'Testing connections...\n',
    daemonLabel: 'Daemon',
    daemonNotReachableSuffix: (url: string) => `not reachable at ${url}`,
    noLlmProviders: 'No LLM providers configured',
    providerLabel: 'Provider',
    modelsSuffix: (n: number) => `${n} models`,
    couldNotFetchProviders: 'Could not fetch providers',
    gatewayLabel: 'Gateway',
    httpError: 'HTTP error',
    notRunningOptional: 'not running (optional)',
    ollamaLabel: 'Ollama',
    ollamaModelsSuffix: (count: number, preview: string, more: boolean) =>
      `${count} models (${preview}${more ? '...' : ''})`,
    notRunning: 'not running',
    configuredSuffix: (n: number) => `${n} configured`,
    contextPrefix: 'ctx:',
    done: 'Done.',
    unknown: 'unknown',
  },
  ko: {
    testingConnections: '연결 테스트 중...\n',
    daemonLabel: 'Daemon',
    daemonNotReachableSuffix: (url: string) => `${url}에서 연결 불가`,
    noLlmProviders: '구성된 LLM provider가 없습니다',
    providerLabel: 'Provider',
    modelsSuffix: (n: number) => `모델 ${n}개`,
    couldNotFetchProviders: 'provider를 가져올 수 없습니다',
    gatewayLabel: 'Gateway',
    httpError: 'HTTP 오류',
    notRunningOptional: '실행 중 아님 (선택 사항)',
    ollamaLabel: 'Ollama',
    ollamaModelsSuffix: (count: number, preview: string, more: boolean) =>
      `모델 ${count}개 (${preview}${more ? '...' : ''})`,
    notRunning: '실행 중 아님',
    configuredSuffix: (n: number) => `${n}개 구성됨`,
    contextPrefix: '컨텍스트:',
    done: '완료.',
    unknown: '알 수 없음',
  },
} as const

// Add a `v` prefix only when the version starts with a digit so a
// daemon reporting `surface-e2e-mock` doesn't render as `vsurface-...`.
function formatVersion(version: string | undefined): string {
  if (!version) return '?'
  return /^\d/.test(version) ? `v${version}` : version
}

interface HealthComponentView {
  status?: string
  details?: string
}

interface OllamaTagsResponse {
  models?: Array<{ name: string }>
}

export async function testConnectionCommand(options: { url?: string }) {
  const locale = detectCliLocale()
  const copy = TEST_CONNECTION_COPY[locale] ?? TEST_CONNECTION_COPY.en
  const client = new DaemonClient(options.url)
  const jsonMode = getOutputFormat() === 'json'

  type Probe = { name: string; status: 'ok' | 'warn' | 'error' | 'absent'; detail?: string; data?: unknown }
  const probes: Probe[] = []

  if (!jsonMode) console.log(chalk.gray(copy.testingConnections))

  // Test daemon
  try {
    const health = await client.health()
    probes.push({
      name: 'daemon',
      status: health.status === 'ok' ? 'ok' : 'warn',
      detail: `${formatVersion(health.version)} (${health.status})`,
      data: health,
    })
    if (!jsonMode) {
      console.log(`${chalk.green('✓')} ${copy.daemonLabel}: ${formatVersion(health.version)} (${health.status})`)
      if (health.components) {
        for (const [name, comp] of Object.entries(health.components) as [string, HealthComponentView][]) {
          const icon = formatHealthComponentIcon(comp.status ?? 'unknown')
          console.log(`  ${icon} ${name}: ${comp.details ?? comp.status}`)
        }
      }
    }
  } catch {
    probes.push({ name: 'daemon', status: 'error', detail: copy.daemonNotReachableSuffix(client.baseUrl) })
    if (jsonMode) {
      console.log(JSON.stringify({ ok: false, probes }, null, 2))
    } else {
      console.log(`${chalk.red('✗')} ${copy.daemonLabel}: ${copy.daemonNotReachableSuffix(client.baseUrl)}`)
    }
    return
  }

  // Test providers
  if (!jsonMode) console.log()
  try {
    const providers = await client.providers()
    probes.push({ name: 'providers', status: providers.length > 0 ? 'ok' : 'warn', detail: copy.configuredSuffix(providers.length), data: providers })
    if (!jsonMode) {
      if (providers.length === 0) console.log(`${chalk.yellow('!')} ${copy.noLlmProviders}`)
      for (const p of providers) {
        console.log(`${chalk.green('✓')} ${copy.providerLabel}: ${p.name} (${p.id}) — ${copy.modelsSuffix(p.models.length)}`)
        for (const m of p.models) {
          console.log(chalk.gray(`    ${m.id} (${copy.contextPrefix} ${m.contextWindow?.toLocaleString() ?? '?'})`))
        }
      }
    }
  } catch {
    probes.push({ name: 'providers', status: 'error', detail: copy.couldNotFetchProviders })
    if (!jsonMode) console.log(`${chalk.red('✗')} ${copy.couldNotFetchProviders}`)
  }

  // Test gateway
  if (!jsonMode) console.log()
  try {
    const gateway = new GatewayClient(resolveGatewayBaseUrl())
    const gatewayHealth = await gateway.healthInfo()
    if (gatewayHealth) {
      probes.push({ name: 'gateway', status: gatewayHealth.status === 'ok' ? 'ok' : 'warn', detail: `${formatVersion(gatewayHealth.version)} (${gatewayHealth.status ?? copy.unknown})`, data: gatewayHealth })
      if (!jsonMode) {
        console.log(`${chalk.green('✓')} ${copy.gatewayLabel}: ${formatVersion(gatewayHealth.version)} (${gatewayHealth.status ?? copy.unknown})`)
        if (gatewayHealth.components) {
          for (const [name, comp] of Object.entries(gatewayHealth.components)) {
            if (!comp) continue
            const icon = formatHealthComponentIcon(comp.status ?? 'unknown')
            console.log(`  ${icon} ${name}: ${comp.details ?? comp.status}`)
          }
        }
      }
    } else {
      probes.push({ name: 'gateway', status: 'warn', detail: copy.httpError })
      if (!jsonMode) console.log(`${chalk.yellow('!')} ${copy.gatewayLabel}: ${copy.httpError}`)
    }
  } catch {
    probes.push({ name: 'gateway', status: 'absent', detail: copy.notRunningOptional })
    if (!jsonMode) console.log(`${chalk.gray('-')} ${copy.gatewayLabel}: ${copy.notRunningOptional}`)
  }

  // Test Ollama directly
  if (!jsonMode) console.log()
  try {
    const res = await fetch('http://localhost:11434/api/tags')
    if (res.ok) {
      const data = await res.json() as OllamaTagsResponse
      const models = data.models?.map((m) => m.name) ?? []
      probes.push({ name: 'ollama', status: 'ok', detail: copy.modelsSuffix(models.length), data: { models } })
      if (!jsonMode) {
        const preview = models.slice(0, 3).join(', ')
        const hasMore = models.length > 3
        console.log(`${chalk.green('✓')} ${copy.ollamaLabel}: ${copy.ollamaModelsSuffix(models.length, preview, hasMore)}`)
      }
    } else {
      probes.push({ name: 'ollama', status: 'absent', detail: copy.notRunning })
    }
  } catch {
    probes.push({ name: 'ollama', status: 'absent', detail: copy.notRunning })
    if (!jsonMode) console.log(`${chalk.gray('-')} ${copy.ollamaLabel}: ${copy.notRunning}`)
  }

  if (jsonMode) {
    const ok = probes.every((p) => p.status === 'ok' || p.status === 'absent')
    console.log(JSON.stringify({ ok, probes }, null, 2))
    return
  }
  console.log(chalk.gray(`\n${copy.done}`))
}
