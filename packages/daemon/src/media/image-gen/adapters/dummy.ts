import type { Provider } from '../adapter.js'

const PNG_1x1_TRANSPARENT = Buffer.from(
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63000100000005000156aa4c2c0000000049454e44ae426082',
  'hex',
)

export const dummyProvider: Provider = {
  info: {
    id: 'dummy',
    label: 'Dummy',
    enabled: true,
    operations: ['text-to-image'],
  },
  async run(input) {
    for (let p = 0.1; p < 1; p += 0.2) {
      await new Promise((r) => setTimeout(r, 10))
      input.onProgress(p)
    }
    input.onProgress(1)
    return {
      outputs: [
        {
          id: `${input.jobId}-0`,
          mime: 'image/png',
          bytes: PNG_1x1_TRANSPARENT,
        },
      ],
    }
  },
}
