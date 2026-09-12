import { Worker } from 'node:worker_threads'
import { cpus } from 'node:os'

interface Task<T> {
  id: string
  fn: string  // Serialized function code
  args: unknown[]
  resolve: (value: T) => void
  reject: (error: Error) => void
}

export class WorkerPool {
  private maxWorkers: number
  private activeWorkers = 0
  private queue: Task<unknown>[] = []

  constructor(maxWorkers?: number) {
    this.maxWorkers = maxWorkers ?? Math.max(1, cpus().length - 1)
  }

  /** Execute a function in a worker thread */
  async execute<Args extends unknown[], T>(
    fn: (...args: Args) => T,
    ...args: Args
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const task: Task<T> = {
        id: Math.random().toString(36).slice(2),
        fn: fn.toString(),
        args,
        resolve: resolve as (value: unknown) => void,
        reject,
      }

      if (this.activeWorkers < this.maxWorkers) {
        this.runTask(task)
      } else {
        this.queue.push(task as Task<unknown>)
      }
    })
  }

  /** Get pool stats */
  getStats(): { active: number; queued: number; maxWorkers: number } {
    return { active: this.activeWorkers, queued: this.queue.length, maxWorkers: this.maxWorkers }
  }

  private runTask<T>(task: Task<T>): void {
    this.activeWorkers++

    // Worker evaluates the serialized function — intentional for sandboxed CPU task execution
    const workerCode = [
      "const { parentPort, workerData } = require('node:worker_threads');",
      "const fn = (0, eval)('(' + workerData.fn + ')');",
      'try {',
      '  const result = fn(...workerData.args);',
      "  if (result && typeof result.then === 'function') {",
      "    result.then(r => parentPort.postMessage({ result: r })).catch(e => parentPort.postMessage({ error: e.message }));",
      '  } else {',
      '    parentPort.postMessage({ result });',
      '  }',
      '} catch (e) {',
      "  parentPort.postMessage({ error: e.message });",
      '}',
    ].join('\n')

    const worker = new Worker(workerCode, {
      eval: true,
      workerData: { fn: task.fn, args: task.args },
    })

    worker.on('message', (msg: { result?: T; error?: string }) => {
      this.activeWorkers--
      if (msg.error) task.reject(new Error(msg.error))
      else task.resolve(msg.result as T)
      this.processQueue()
      worker.terminate()
    })

    worker.on('error', (err: Error) => {
      this.activeWorkers--
      task.reject(err)
      this.processQueue()
    })
  }

  private processQueue(): void {
    while (this.queue.length > 0 && this.activeWorkers < this.maxWorkers) {
      const next = this.queue.shift()
      if (next) this.runTask(next)
    }
  }
}
