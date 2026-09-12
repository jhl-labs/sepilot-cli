import { defineConfig } from 'tsup'

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/apps.ts',
    'src/node.ts',
    'src/desktop.ts',
    'src/http.ts',
    'src/extension.ts',
    'src/launcher/index.ts',
    'src/daemon/http.ts',
    'src/daemon/types.ts',
    'src/daemon/image-canvas.ts',
    'src/daemon/chat-surface.ts',
    'src/daemon/chat-transport.ts',
    'src/daemon/stream.ts',
    'src/daemon/ws.ts',
    'src/gateway/http.ts',
  ],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: false,
  minify: true,
  clean: false,
})
