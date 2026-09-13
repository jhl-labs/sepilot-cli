import { createHash, randomBytes } from 'node:crypto'
/** Single-use bootstrap tickets. Never persisted and invalidated by token rotation. */
export class MobilePairingTickets {
  private tickets = new Map<string, { expiresAt: number; tokenHash: string }>()
  constructor(private readonly now = Date.now) {}
  private hash(value: string) { return createHash('sha256').update(value).digest('hex') }
  issue(token: string) {
    for (const [key, ticket] of this.tickets) if (ticket.expiresAt <= this.now()) this.tickets.delete(key)
    if (this.tickets.size >= 5) this.tickets.delete(this.tickets.keys().next().value!)
    const code = randomBytes(16).toString('base64url')
    const expiresAt = this.now() + 120_000
    this.tickets.set(this.hash(code), { expiresAt, tokenHash: this.hash(token) })
    return { code, expiresAt }
  }
  consume(code: string, token: string) {
    const key = this.hash(code)
    const ticket = this.tickets.get(key)
    if (!ticket) return false
    this.tickets.delete(key)
    return ticket.expiresAt > this.now() && ticket.tokenHash === this.hash(token)
  }
}
