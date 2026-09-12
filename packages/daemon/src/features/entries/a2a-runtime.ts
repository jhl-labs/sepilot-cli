import { join } from 'node:path'
import { A2AFileTaskStore } from '../../a2a/task-store.js'
import { A2AHttpPushDelivery } from '../../a2a/push.js'
import { createA2ASendTool } from '../../tools/a2a-send.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type { A2ATaskStore } from '../../a2a/server.js'
import type { FeatureRuntimeToolDeps } from '../types.js'

/**
 * Creates the a2a task store, registers the `a2a.send` tool, and returns the
 * task store so buildRuntime can expose it on the runtime object. Only imported
 * (and thus only bundled) when the `a2a` feature is enabled.
 */
export function createA2aRuntime(
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
): A2ATaskStore {
  const taskStore = new A2AFileTaskStore(join(deps.dataDir, 'sessions', 'a2a', 'tasks.json'), {
    pushDelivery: new A2AHttpPushDelivery(),
  })
  registry.register(createA2ASendTool())
  return taskStore
}
