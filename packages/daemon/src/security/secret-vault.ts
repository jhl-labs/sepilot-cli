import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { EncryptionManager } from './encryption.js'
import { createLogger } from '../logger.js'

const log = createLogger('security.vault')
const SECRET_DIR_MODE = 0o700
const SECRET_FILE_MODE = 0o600

export interface SecretVaultOptions {
  allowPlaintextSecrets?: boolean
}

export class SecretVault {
  private secrets = new Map<string, string>()
  private encPath: string
  private plainPath: string

  constructor(
    private baseDir: string,
    private encryption: EncryptionManager,
    private options: SecretVaultOptions = {},
  ) {
    this.encPath = join(baseDir, 'secrets.enc')
    this.plainPath = join(baseDir, 'secrets.json')
  }

  async load(): Promise<void> {
    await this.ensureBaseDir()
    if (this.encryption.isEnabled()) {
      try {
        const data = await readFile(this.encPath, 'utf-8')
        const json = this.encryption.decrypt(data)
        const parsed = JSON.parse(json) as Record<string, string>
        this.secrets = new Map(Object.entries(parsed))
        return
      } catch { /* file doesn't exist or decrypt failed — start fresh */ }
    }
    // Plaintext fallback
    try {
      const data = await readFile(this.plainPath, 'utf-8')
      const parsed = JSON.parse(data) as Record<string, string>
      this.secrets = new Map(Object.entries(parsed))
      if (!this.encryption.isEnabled()) {
        log.warn('Secrets loaded from plaintext — encryption key not configured')
      }
    } catch { /* no file — empty */ }
  }

  async set(key: string, value: string): Promise<void> {
    this.secrets.set(key, value)
    await this.persist()
  }

  get(key: string): string | null {
    return this.secrets.get(key) ?? null
  }

  list(): string[] {
    return Array.from(this.secrets.keys())
  }

  async remove(key: string): Promise<boolean> {
    if (!this.secrets.has(key)) return false
    this.secrets.delete(key)
    await this.persist()
    return true
  }

  private async persist(): Promise<void> {
    await this.ensureBaseDir()
    const json = JSON.stringify(Object.fromEntries(this.secrets))
    if (this.encryption.isEnabled()) {
      const encrypted = this.encryption.encrypt(json)
      await this.writeSecretFile(this.encPath, encrypted)
    } else {
      if (!this.options.allowPlaintextSecrets) {
        throw new Error(
          'Refusing to persist secrets as plaintext because encryption is disabled. Set memory.allowPlaintextSecrets=true to opt in.',
        )
      }
      log.warn('Persisting secrets as plaintext — encryption key not configured')
      await this.writeSecretFile(this.plainPath, json)
    }
  }

  private async ensureBaseDir(): Promise<void> {
    await mkdir(this.baseDir, { recursive: true, mode: SECRET_DIR_MODE })
    await chmod(this.baseDir, SECRET_DIR_MODE).catch(() => {})
  }

  private async writeSecretFile(path: string, content: string): Promise<void> {
    await writeFile(path, content, { encoding: 'utf-8', mode: SECRET_FILE_MODE })
    await chmod(path, SECRET_FILE_MODE).catch(() => {})
  }
}
