import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { readFile } from 'node:fs/promises'

const ALGORITHM = 'aes-256-gcm'
const NONCE_LENGTH = 12
const TAG_LENGTH = 16
const MAGIC = Buffer.from('SEPD') // 4-byte file magic

export class EncryptionManager {
  private key: Buffer | null = null

  async loadKey(keyPath: string): Promise<void> {
    this.key = await readFile(keyPath)
    if (this.key.length !== 32) {
      throw new Error(`Invalid key length: expected 32 bytes, got ${this.key.length}`)
    }
  }

  isEnabled(): boolean {
    return this.key !== null
  }

  /** Encrypt a string, return base64-encoded ciphertext */
  encrypt(plaintext: string): string {
    if (!this.key) throw new Error('Encryption key not loaded')
    const nonce = randomBytes(NONCE_LENGTH)
    const cipher = createCipheriv(ALGORITHM, this.key, nonce)
    const encrypted = Buffer.concat([cipher.update(plaintext, 'utf-8'), cipher.final()])
    const tag = cipher.getAuthTag()
    // Format: base64(nonce + ciphertext + tag)
    return Buffer.concat([nonce, encrypted, tag]).toString('base64')
  }

  /** Decrypt a base64-encoded ciphertext */
  decrypt(ciphertext: string): string {
    if (!this.key) throw new Error('Encryption key not loaded')
    const data = Buffer.from(ciphertext, 'base64')
    const nonce = data.subarray(0, NONCE_LENGTH)
    const tag = data.subarray(data.length - TAG_LENGTH)
    const encrypted = data.subarray(NONCE_LENGTH, data.length - TAG_LENGTH)
    const decipher = createDecipheriv(ALGORITHM, this.key, nonce)
    decipher.setAuthTag(tag)
    return decipher.update(encrypted) + decipher.final('utf-8')
  }

  /** Encrypt a file (returns buffer with SEPD magic + version + nonce + ciphertext + tag) */
  encryptFile(content: string): Buffer {
    if (!this.key) throw new Error('Encryption key not loaded')
    const nonce = randomBytes(NONCE_LENGTH)
    const cipher = createCipheriv(ALGORITHM, this.key, nonce)
    const encrypted = Buffer.concat([cipher.update(content, 'utf-8'), cipher.final()])
    const tag = cipher.getAuthTag()
    const version = Buffer.from([1]) // version byte
    return Buffer.concat([MAGIC, version, nonce, encrypted, tag])
  }

  /** Decrypt a file buffer */
  decryptFile(data: Buffer): string {
    if (!this.key) throw new Error('Encryption key not loaded')
    // Verify magic
    if (data.subarray(0, 4).toString() !== 'SEPD') {
      throw new Error('Invalid encrypted file format')
    }
    const version = data[4]
    if (version !== 1) throw new Error(`Unsupported encryption version: ${version}`)
    const nonce = data.subarray(5, 5 + NONCE_LENGTH)
    const tag = data.subarray(data.length - TAG_LENGTH)
    const encrypted = data.subarray(5 + NONCE_LENGTH, data.length - TAG_LENGTH)
    const decipher = createDecipheriv(ALGORITHM, this.key, nonce)
    decipher.setAuthTag(tag)
    return decipher.update(encrypted) + decipher.final('utf-8')
  }
}
