import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { readFile } from 'node:fs/promises'

const ALGORITHM = 'aes-256-gcm'
const NONCE_LENGTH = 12
const TAG_LENGTH = 16
const MAGIC = Buffer.from('SEPD')

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

  encrypt(plaintext: string): string {
    if (!this.key) throw new Error('Encryption key not loaded')
    const nonce = randomBytes(NONCE_LENGTH)
    const cipher = createCipheriv(ALGORITHM, this.key, nonce)
    const encrypted = Buffer.concat([cipher.update(plaintext, 'utf-8'), cipher.final()])
    const tag = cipher.getAuthTag()
    return Buffer.concat([nonce, encrypted, tag]).toString('base64')
  }

  decrypt(ciphertext: string): string {
    if (!this.key) throw new Error('Encryption key not loaded')
    const data = Buffer.from(ciphertext, 'base64')
    assertSealedLength(data.length, NONCE_LENGTH)
    const nonce = data.subarray(0, NONCE_LENGTH)
    const tag = data.subarray(data.length - TAG_LENGTH)
    const encrypted = data.subarray(NONCE_LENGTH, data.length - TAG_LENGTH)
    const decipher = createDecipheriv(ALGORITHM, this.key, nonce, { authTagLength: TAG_LENGTH })
    decipher.setAuthTag(tag)
    return decipher.update(encrypted) + decipher.final('utf-8')
  }

  encryptFile(content: string): Buffer {
    if (!this.key) throw new Error('Encryption key not loaded')
    const nonce = randomBytes(NONCE_LENGTH)
    const cipher = createCipheriv(ALGORITHM, this.key, nonce)
    const encrypted = Buffer.concat([cipher.update(content, 'utf-8'), cipher.final()])
    const tag = cipher.getAuthTag()
    const version = Buffer.from([1])
    return Buffer.concat([MAGIC, version, nonce, encrypted, tag])
  }

  decryptFile(data: Buffer): string {
    if (!this.key) throw new Error('Encryption key not loaded')
    if (data.subarray(0, 4).toString() !== 'SEPD') {
      throw new Error('Invalid encrypted file format')
    }
    const version = data[4]
    if (version !== 1) throw new Error(`Unsupported encryption version: ${version}`)
    assertSealedLength(data.length, 5 + NONCE_LENGTH)
    const nonce = data.subarray(5, 5 + NONCE_LENGTH)
    const tag = data.subarray(data.length - TAG_LENGTH)
    const encrypted = data.subarray(5 + NONCE_LENGTH, data.length - TAG_LENGTH)
    const decipher = createDecipheriv(ALGORITHM, this.key, nonce, { authTagLength: TAG_LENGTH })
    decipher.setAuthTag(tag)
    return decipher.update(encrypted) + decipher.final('utf-8')
  }
}

function assertSealedLength(length: number, headerLength: number): void {
  if (length < headerLength + TAG_LENGTH) throw new Error('Invalid encrypted payload')
}
