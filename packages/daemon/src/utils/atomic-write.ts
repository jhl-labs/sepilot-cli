import { writeFile, rename, open, rm } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'
import { secureFileAsync } from './secure-file.js'
import { retryTransientFsOperation, type TransientFsRetryOptions } from './fs-retry.js'

const shouldFsync = () => process.env.SEPILOTD_CHECKPOINT_FSYNC === '1'

export async function writeFileAtomic(
  path: string,
  data: string,
  opts?: { fsync?: boolean; renameRetry?: TransientFsRetryOptions },
): Promise<void> {
  const tmp = `${path}.${process.pid}.${randomUUID()}.tmp`
  try {
    await writeFile(tmp, data, {
      encoding: 'utf-8',
      flag: 'wx',
      mode: 0o600,
    })
    if (opts?.fsync ?? shouldFsync()) {
      const fh = await open(tmp, 'r+')
      try {
        await fh.sync()
      } finally {
        await fh.close()
      }
    }
    await retryTransientFsOperation(() => rename(tmp, path), opts?.renameRetry)
  } catch (err) {
    await retryTransientFsOperation(() => rm(tmp, { force: true }), opts?.renameRetry).catch(
      () => undefined,
    )
    throw err
  }
  await secureFileAsync(path)
}
