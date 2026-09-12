export interface LspServerSpec {
  language: string
  bin: string
  args: string[]
  fileExtensions: string[]
}

export const DEFAULT_LSP_SERVERS: readonly LspServerSpec[] = Object.freeze([
  {
    language: 'typescript',
    bin: 'typescript-language-server',
    args: ['--stdio'],
    fileExtensions: ['.ts', '.tsx', '.js', '.jsx'],
  },
  {
    language: 'python',
    bin: 'pyright-langserver',
    args: ['--stdio'],
    fileExtensions: ['.py'],
  },
  {
    language: 'rust',
    bin: 'rust-analyzer',
    args: [],
    fileExtensions: ['.rs'],
  },
  {
    language: 'go',
    bin: 'gopls',
    args: [],
    fileExtensions: ['.go'],
  },
])

export interface DiscoverLspDeps {
  which: (bin: string) => Promise<string | null>
}

export async function discoverLspServers(
  deps: DiscoverLspDeps,
  candidates: readonly LspServerSpec[] = DEFAULT_LSP_SERVERS,
): Promise<LspServerSpec[]> {
  const out: LspServerSpec[] = []
  for (const spec of candidates) {
    const path = await deps.which(spec.bin)
    if (path) out.push(spec)
  }
  return out
}
