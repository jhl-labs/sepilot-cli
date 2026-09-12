import type { FastifyInstance } from 'fastify'
import { join } from 'node:path'
import { bindCapability } from '../capabilities/bind.js'
import { sepilotdHome } from '../../storage/home.js'
import { DAEMON_VERSION } from '../../version.js'

interface EnvKnob {
  /** Environment variable name. */
  name: string
  /** Currently effective value (env or default). */
  effective: string | null
  /** Whether the value comes from the env (true) or from the default (false). */
  fromEnv: boolean
  /** Documented default applied when env is unset, or null if there is none. */
  default: string | null
  /** Human-readable purpose for the System tab. */
  purpose: string
}

interface PathEntry {
  /** Display label, e.g. "sessions". */
  label: string
  /** Absolute path on disk. */
  path: string
}

interface BootInfo {
  daemon: {
    version: string
    nodeVersion: string
    pid: number
    platform: string
    uptimeSeconds: number
    startedAt: string
  }
  paths: {
    home: string
    entries: PathEntry[]
  }
  env: EnvKnob[]
}

/**
 * Capture the current value of an env knob, treating an empty string as
 * unset (Unix convention).
 */
function readEnv(name: string, fallback: string | null, purpose: string): EnvKnob {
  const raw = process.env[name]
  const set = typeof raw === 'string' && raw.length > 0
  return {
    name,
    fromEnv: set,
    effective: set ? (raw as string) : fallback,
    default: fallback,
    purpose,
  }
}

const startedAtEpoch = Date.now()

/**
 * `GET /system/boot-info` — read-only snapshot of values that are set at
 * daemon boot via env vars / config. The desktop's System tab uses this
 * so users can see what is actually applied even when the GUI cannot
 * change it (port, storage path, streaming knobs).
 */
export async function registerSystemBootInfoRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'system-boot-info',
      version: '1',
      methods: [{ method: 'GET', path: '/system/boot-info' }],
    },
    async (a) => {
      a.get('/system/boot-info', async () => {
        const home = sepilotdHome()
        const info: BootInfo = {
          daemon: {
            version: DAEMON_VERSION,
            nodeVersion: process.version,
            pid: process.pid,
            platform: `${process.platform}/${process.arch}`,
            uptimeSeconds: Math.floor(process.uptime()),
            startedAt: new Date(startedAtEpoch).toISOString(),
          },
          paths: {
            home,
            entries: [
              { label: 'sessions', path: join(home, 'sessions') },
              { label: 'memory', path: join(home, 'memory') },
              { label: 'skills', path: join(home, 'skills') },
              { label: 'security', path: join(home, 'security') },
              { label: 'config.yaml', path: join(home, 'config.yaml') },
            ],
          },
          env: [
            readEnv(
              'SEPILOTD_DATA_DIR',
              null,
              'canonical daemon profile root. Embedded callers may set the same value through startup options.',
            ),
            readEnv(
              'SEPILOTD_HOME',
              null,
              'explicit legacy domain-storage override. Usually leave unset so DATA_DIR owns one profile.',
            ),
            readEnv(
              'SEPILOTD_HTTP_PORT',
              '17600',
              'daemon HTTP/WebSocket 리슨 포트.',
            ),
            readEnv(
              'SEPILOTD_HTTP_HOST',
              '127.0.0.1',
              'daemon 바인딩 호스트. 외부 접속 허용 시 변경.',
            ),
            readEnv(
              'SEPILOTD_GATEWAY_PORT',
              '17610',
              'gateway 사이드카 포트.',
            ),
            readEnv(
              'SEPILOTD_AGENT_INACTIVITY_MS',
              '300000',
              'agent loop 정적 타임아웃. 초과 시 SSE에 AGENT_INACTIVITY 전송.',
            ),
            readEnv(
              'SEPILOTD_SSE_KEEPALIVE_MS',
              '15000',
              'SSE keepalive 주기. cli stream-idle watchdog과 짝.',
            ),
            readEnv(
              'SEPILOTD_LARGE_CODEBASE_SCOUT',
              'off',
              'coder graph의 planning 전 read-only scout 실행. 1이면 활성.',
            ),
            readEnv(
              'SEPILOTD_LARGE_CODEBASE_SCOUT_MAX_PROMPTS',
              '2',
              '대용량 코드베이스 scout subagent dispatch 최대 개수.',
            ),
            readEnv(
              'SEPILOTD_LARGE_CODEBASE_SCOUT_MAX_ITERATIONS',
              '12',
              '각 대용량 코드베이스 scout subagent의 최대 반복 수.',
            ),
            readEnv(
              'SEPILOTD_LOG_LEVEL',
              'info',
              'pino 로그 레벨 (trace/debug/info/warn/error).',
            ),
          ],
        }
        return info
      })
    },
  )
}
