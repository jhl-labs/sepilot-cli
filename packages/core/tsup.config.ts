import { defineConfig } from 'tsup'

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/canvas/parse.ts',
    'src/canvas/image.ts',
  ],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: false,
  minify: true,
  clean: true,
})
