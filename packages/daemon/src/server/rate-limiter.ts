import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify'
import { skipOverride } from './skip-override.js'

interface Bucket { tokens: number; lastRefill: number }

export class DaemonRateLimiter {
  private ipBuckets = new Map<string, Bucket>()
  private rps: number
  private burst: number

  constructor(rps = 50, burst = 100) {
    this.rps = rps
    this.burst = burst
  }

  check(ip: string): boolean {
    if (!this.ipBuckets.has(ip)) {
      this.ipBuckets.set(ip, { tokens: this.burst, lastRefill: Date.now() })
    }
    const bucket = this.ipBuckets.get(ip)!
    const now = Date.now()
    const elapsed = (now - bucket.lastRefill) / 1000
    bucket.tokens = Math.min(this.burst, bucket.tokens + elapsed * this.rps)
    bucket.lastRefill = now
    if (bucket.tokens < 1) return false
    bucket.tokens--
    return true
  }

  cleanup(): void {
    const cutoff = Date.now() - 120000
    for (const [ip, b] of this.ipBuckets) { if (b.lastRefill < cutoff) this.ipBuckets.delete(ip) }
  }

  reset(): void {
    this.ipBuckets.clear()
  }
}

export async function daemonRateLimitPlugin(app: FastifyInstance) {
  const limiter = new DaemonRateLimiter()
  const timer = setInterval(() => limiter.cleanup(), 60000)
  timer.unref?.()
  app.addHook('onClose', () => clearInterval(timer))
  app.decorate('daemonRateLimiter', limiter)

  app.addHook('onRequest', async (request: FastifyRequest, reply: FastifyReply) => {
    if (request.method === 'OPTIONS') return
    if (request.url === '/api/v1/health') return
    if (!limiter.check(request.ip)) {
      reply.header('Retry-After', '1')
      return reply.status(429).send({ error: { code: 'RATE_LIMITED', message: 'Too many requests' } })
    }
  })
}

skipOverride(daemonRateLimitPlugin)
