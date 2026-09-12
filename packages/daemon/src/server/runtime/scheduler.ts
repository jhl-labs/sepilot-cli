import type { SchedulerStack } from './services.js'

export function configureScheduler(stack: SchedulerStack): void {
  if (process.env.SEPILOTD_SCHEDULER_DISABLED === '1') return
  stack.engine.start()
  stack.deliveryWorker?.start()
}
