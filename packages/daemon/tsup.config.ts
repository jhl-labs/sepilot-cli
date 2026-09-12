import { defineConfig } from 'tsup'

const isRelease =
  process.env.SEPILOTD_RELEASE_BUILD === '1' || process.env.NODE_ENV === 'production'

// Release 빌드에서는 daemon 전체를 단일 번들로 inline해 portable 추출 시 파일 수를
// 격감시킨다(9k+ → 수십). Windows에서 추출/Defender 스캔 비용이 byte보다 file-count에
// 비례하므로 가장 큰 효과. external에 둘 것:
//   - native binding(.node 파일 동적 로드)
//   - require.resolve로 자기 디렉토리 내부 asset을 읽는 패키지
//   - 사용자 plugin/dynamic 로딩 대상
// native bindings — bundling 불가. release/dev 양쪽 모두 external 이어야 한다.
// (ssh2/cpu-features 는 .node 가 optional 이라 미빌드 환경에선 bundler 가 resolve
//  실패로 죽는다.) dockerode/docker-modem 은 native 가 아니지만 eager 하게
// require('ssh2') 하므로, inline 하면 번들 위치(dist/)에서 ssh2 를 못 찾아
// 런타임에 죽는다 — external 인 ssh2 를 부르는 패키지도 같이 external 로 둬서
// node_modules 에서 resolve 되게 한다.
const nativeBindings = [
  'node-pty',
  'better-sqlite3',
  /^sqlite-vec/,
  'bufferutil',
  'utf-8-validate',
  'cpu-features',
  'ssh2',
  'dockerode',
  'docker-modem',
]

const releaseExternal = [
  ...nativeBindings,
  // 자기 디렉토리에서 worker/fonts/asset을 fs.readFile로 로드하는 패키지
  'pdf-parse',
  'pdfjs-dist',
  // 동적/런타임 로딩
  'playwright',
  'pino-pretty',
  // node-telegram-bot-api는 내부에서 file path 기반 asset/local request 모듈 사용
  'node-telegram-bot-api',
]

export default defineConfig((options) => {
  const isWatch = Boolean(options.watch)

  return {
    entry: ['src/index.ts', 'src/benchmark.ts', 'src/embedded.ts', 'src/main.ts'],
    format: ['esm'],
    // dev: dts 끔 — daemon 코드 양 때문에 dts 빌드가 분 단위라 onSuccess
    //       트리거가 너무 늦어 tsup --watch 사이클이 망가짐. release만 dts 켬
    //       (packages/bundle이 .d.ts를 import).
    // build: workspace consumers resolve package exports to dist/*.d.ts during
    //        typecheck, so non-watch builds must emit declarations.
    // dev: minify 끔 — dev cycle 단축 + 디버깅 가독성.
    dts: isRelease || !isWatch,
    sourcemap: false,
    minify: isRelease,
    treeshake: isRelease,
    splitting: !isRelease, // release는 단일 번들 (splitting=false)
    clean: true,
    // ESM 번들에서 inline된 CJS 패키지가 dynamic require를 호출해도 동작하도록
    // createRequire shim만 inject한다. __dirname / __filename은 esbuild가 ESM
    // 번들 시 자체적으로 `var __dirname = dirname(fileURLToPath(import.meta.url))`
    // 패턴으로 자동 inject하므로 여기서 또 선언하면 release-hardening 단계의
    // esbuild minify가 "symbol __dirname has already been declared"로 fail한다.
    // (2026-05-17 회귀 — docs/troubleshooting/desktop-daemon-connection.md §6.)
    banner: {
      js: [
        '#!/usr/bin/env node',
        "import { createRequire as __sepilotCreateRequire } from 'module';",
        'const require = __sepilotCreateRequire(import.meta.url);',
      ].join('\n'),
    },
    external: isRelease ? releaseExternal : ['playwright', 'yaml', ...nativeBindings],
    // releaseExternal에 없는 모든 패키지를 inline 강제
    noExternal: isRelease
      ? [
          /^(?!node-pty$|better-sqlite3$|sqlite-vec|bufferutil$|utf-8-validate$|cpu-features$|ssh2$|dockerode$|docker-modem$|pdf-parse$|pdfjs-dist$|playwright$|pino-pretty$|node-telegram-bot-api$|^node:).+/,
        ]
      : [],
  }
})
