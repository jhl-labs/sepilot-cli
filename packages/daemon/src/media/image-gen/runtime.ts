import { dummyProvider } from './adapters/dummy.js'
import { codexImageGenProvider } from './adapters/codex.js'
import { localPythonProvider } from './adapters/local-python.js'
import { createQueue, type ImageGenQueue } from './queue.js'

export const imageGenProviders = new Map([
  [dummyProvider.info.id, dummyProvider],
  [localPythonProvider.info.id, localPythonProvider],
  [codexImageGenProvider.info.id, codexImageGenProvider],
])

let imageGenQueueSingleton: ImageGenQueue | null = null

export function getImageGenQueue(): ImageGenQueue {
  if (!imageGenQueueSingleton) {
    imageGenQueueSingleton = createQueue(imageGenProviders)
    imageGenQueueSingleton.start()
  }
  return imageGenQueueSingleton
}
