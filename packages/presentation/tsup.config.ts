import { defineConfig } from 'tsup'

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/canvas/react.tsx',
    'src/artifacts/react.tsx',
    'src/artifacts/shared.ts',
    'src/chat/composer-shortcuts.ts',
  ],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: false,
  minify: true,
  clean: true,
  external: [
    '@sepilotd/core',
    'react',
    'react-dom',
    'react/jsx-runtime',
  ],
})
