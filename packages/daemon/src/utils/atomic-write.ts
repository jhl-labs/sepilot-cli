import { writeFile, rename, open, rm } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'
import { secureFileAsync } from './secure-file.js'
import { retryTransientFsOperation, type TransientFsRetryOptions } from './fs-retry.js'

/**
 * `fsync` is off by default — the extra fd flush is a measurable throughput
 * regression on hot checkpoint paths. Deployments that value crash durability
 * over write latency opt in with `SEPILOTD_CHECKPOINT_FSYNC=1`, or a caller can
 * force it per-write via the `fsync` option.
 */
const shouldFsync = () => process.env.SEPILOTD_CHECKPOINT_FSYNC === '1'

/**
 * Write `data` to `path` atomically: stage in a unique sibling temp file and
 * `rename` it into place. A concurrent reader sees either the previous whole
 * file or the new whole file — never a half-written checkpoint. File mode is
 * kept at 0600 from staging onward.
 */
export async function writeFileAtomic(
  path: string,
  data: string,
  opts?: { fsync?: boolean; renameRetry?: TransientFsRetryOptions },
): Promise<void> {
  // A PID alone is not unique when independent stores write concurrently in
  // the same daemon process. Give every transaction its own staging file so
  // one rename cannot consume another writer's payload.
  const tmp = `${path}.${process.pid}.${randomUUID()}.tmp`
  try {
    await writeFile(tmp, data, {
      encoding: 'utf-8',
      flag: 'wx',
      mode: 0o600,
    })
    if (opts?.fsync ?? shouldFsync()) {
      // Windows rejects fsync on a read-only handle with EPERM. The staging
      // file is private and complete, so open it read/write solely for sync.
      const fh = await open(tmp, 'r+')
      try {
        await fh.sync()
      } finally {
        await fh.close()
      }
    }
    // Windows can reject replacement for a short period while antivirus,
    // indexing, backup, or another reader holds a sharing lock. Retry the
    // already-complete staging file; never delete the known-good destination
    // as a fallback because that would create a data-loss window.
    await retryTransientFsOperation(() => rename(tmp, path), opts?.renameRetry)
  } catch (err) {
    // Cleanup is best-effort and must not mask the original write/rename error.
    await retryTransientFsOperation(() => rm(tmp, { force: true }), opts?.renameRetry).catch(
      () => undefined,
    )
    throw err
  }
  await secureFileAsync(path)
}
