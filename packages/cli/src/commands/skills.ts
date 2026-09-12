import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const SKILLS_COPY = {
  en: {
    noSkillsInstalled: 'No skills installed.',
    disabled: '✗ disabled',
    enabled: '✓ enabled ',
    autoManaged: '[A] auto (edit SKILL.md)',
    sourcePrefix: 'source:',
    toolsPrefix: 'tools:',
    none: '(none)',
    skillEnabledPrefix: (id: string) => `Skill enabled: ${id}`,
    skillDisabledPrefix: (id: string) => `Skill disabled: ${id}`,
    pptxAuthorNote:
    'Note: this skill writes .pptx via `python-pptx` (or Marp / reveal-md for Markdown→slides). Install with `uv pip install python-pptx` (or equivalent). For corporate templates, point the agent at the .potx / template .pptx so layouts inherit.',
    docxAuthorNote:
    'Note: this skill writes .docx via `python-docx` (or `pandoc -t docx --reference-doc=...` for Markdown source). Install with `uv pip install python-docx` (or `brew install pandoc`).',
    xlsxAuthorNote:
    'Note: this skill writes .xlsx via `openpyxl` (or `xlsxwriter` / `pandas`). Install with `uv pip install openpyxl pandas`. Always prefers formulas over hard-coded values for auditability.',
    noMarketplaceSkills: 'No marketplace skills found.',
    installed: 'installed',
    notInstalled: 'not installed',
    installHint: (source: string) => `    install: sepilot skills install ${source}`,
    noMatchingSkills: 'No matching skills.',
    hiddenBuiltinsHint: (n: number) =>
    `\n${n} built-in skill${n === 1 ? '' : 's'} hidden. Run \`sepilot skills list --include-disabled\` to see them, or \`sepilot skills enable <id>\` to activate.`,
  },
  ko: {
    noSkillsInstalled: '설치된 스킬이 없습니다.',
    disabled: '✗ 비활성화됨',
    enabled: '✓ 활성화됨 ',
    autoManaged: '[A] 자동 발견 (SKILL.md에서 관리)',
    sourcePrefix: '소스:',
    toolsPrefix: '도구:',
    none: '(없음)',
    skillEnabledPrefix: (id: string) => `스킬 활성화됨: ${id}`,
    skillDisabledPrefix: (id: string) => `스킬 비활성화됨: ${id}`,
    pptxAuthorNote:
    '참고: 이 스킬은 `python-pptx`로 .pptx를 작성합니다 (Markdown→슬라이드는 Marp / reveal-md). `uv pip install python-pptx` 등으로 설치하세요. 사내 템플릿이 있으면 .potx / 기존 .pptx 경로를 알려주면 layout을 상속합니다.',
    docxAuthorNote:
    '참고: 이 스킬은 `python-docx`로 .docx를 작성합니다 (Markdown 원본은 `pandoc -t docx --reference-doc=...`). `uv pip install python-docx` (또는 `brew install pandoc`)로 설치하세요.',
    xlsxAuthorNote:
    '참고: 이 스킬은 `openpyxl` (또는 `xlsxwriter` / `pandas`)로 .xlsx를 작성합니다. `uv pip install openpyxl pandas`로 설치하세요. 감사 가능성을 위해 하드코딩 값보다 수식을 우선합니다.',
    noMarketplaceSkills: '마켓플레이스 스킬을 찾을 수 없습니다.',
    installed: '설치됨',
    notInstalled: '설치 안 됨',
    installHint: (source: string) => `    설치: sepilot skills install ${source}`,
    noMatchingSkills: '일치하는 스킬이 없습니다.',
    hiddenBuiltinsHint: (n: number) =>
    `\n빌트인 스킬 ${n}개가 숨겨져 있습니다. 보려면 \`sepilot skills list --include-disabled\`, 활성화하려면 \`sepilot skills enable <id>\`를 사용하세요.`,
  },
} as const

function skillsCopy() {
  return SKILLS_COPY[detectCliLocale()] ?? SKILLS_COPY.en
}

type EnableNoteKey = keyof typeof SKILLS_COPY.en

const ENABLE_NOTES: Record<string, EnableNoteKey> = {
  'pptx-author': 'pptxAuthorNote',
  'docx-author': 'docxAuthorNote',
  'xlsx-author': 'xlsxAuthorNote',
}

type SkillsListClient = Pick<DaemonClient, 'skills'>
type SkillsToggleClient = Pick<DaemonClient, 'setSkillEnabled'>

export async function skillsCommandImpl(
  client: SkillsListClient,
  opts: { includeDisabled?: boolean; cwd?: string; workspaceRoot?: string } = {},
): Promise<void> {
  const copy = skillsCopy()
  const context = {
    cwd: opts.cwd ?? process.cwd(),
    workspaceRoot: opts.workspaceRoot ?? opts.cwd ?? process.cwd(),
  }
  const data = await client.skills({ includeDisabled: opts.includeDisabled, ...context })

  let hiddenCount = 0
  if (!opts.includeDisabled) {
    const all = await client.skills({ includeDisabled: true, ...context })
    hiddenCount = Math.max(0, (all?.length ?? 0) - (data?.length ?? 0))
  }

  output(data ?? [], (skills) => {
    const body = skills.length
      ? skills
          .map((s) => {
            const state = s.autoDiscovered
              ? chalk.cyan(copy.autoManaged)
              : s.enabled === false
                ? chalk.gray(copy.disabled)
                : chalk.green(copy.enabled)
            const source = s.source ? `\n    ${copy.sourcePrefix} ${s.source.type}:${s.source.ref}` : ''
            return `  ${state}  ${s.name.padEnd(24)} v${s.version.padEnd(7)} ${s.description}\n    ${copy.toolsPrefix} ${s.tools.join(', ') || copy.none}${source}`
          })
          .join('\n\n')
      : copy.noSkillsInstalled

    return hiddenCount > 0
      ? `${body}\n${chalk.gray(copy.hiddenBuiltinsHint(hiddenCount))}`
      : body
  })
}

export async function skillEnableCommandImpl(id: string, client: SkillsToggleClient): Promise<void> {
  const copy = skillsCopy()
  const result = await client.setSkillEnabled(id, true)
  console.log(copy.skillEnabledPrefix(result.id))
  const noteKey = ENABLE_NOTES[id]
  if (noteKey) {
    const note = copy[noteKey]
    if (typeof note === 'string') console.log(note)
  }
}

export async function skillDisableCommandImpl(id: string, client: SkillsToggleClient): Promise<void> {
  const copy = skillsCopy()
  const result = await client.setSkillEnabled(id, false)
  console.log(copy.skillDisabledPrefix(result.id))
}

export async function skillsCommand(options: { url?: string; includeDisabled?: boolean } = {}): Promise<void> {
  await skillsCommandImpl(new DaemonClient(options.url), { includeDisabled: options.includeDisabled })
}

export async function skillEnableCommand(id: string, options: { url?: string } = {}): Promise<void> {
  const client = new DaemonClient(options.url)
  await assertManagedSkill(client, id)
  await skillEnableCommandImpl(id, client)
}

export async function skillDisableCommand(id: string, options: { url?: string } = {}): Promise<void> {
  const client = new DaemonClient(options.url)
  await assertManagedSkill(client, id)
  await skillDisableCommandImpl(id, client)
}

async function assertManagedSkill(client: DaemonClient, id: string): Promise<void> {
  const cwd = process.cwd()
  const detail = await client.skill(id, { cwd, workspaceRoot: cwd })
  if (detail.metadata.autoDiscovered) {
    throw new Error(
      `Skill "${id}" is auto-discovered and cannot be enabled or disabled. Edit its SKILL.md instead.`,
    )
  }
}

export async function skillSearchCommand(
  query: string,
  options: { url?: string; remote?: boolean; marketplace?: string; limit?: string },
) {
  const copy = skillsCopy()
  const client = new DaemonClient(options.url)
  if (options.remote) {
    const limit = options.limit ? Number.parseInt(options.limit, 10) : undefined
    const data = await client.searchMarketplaceSkills(query, {
      marketplace: options.marketplace,
      limit: Number.isFinite(limit) ? limit : undefined,
    })
    output(data ?? [], (skills) => {
      if (!skills.length) return copy.noMarketplaceSkills
      return skills.map((s) => {
        const installed = s.installed ? chalk.green(copy.installed) : chalk.gray(copy.notInstalled)
        return [
          `  ${chalk.bold(s.metadata.name)} v${s.metadata.version} ${chalk.gray(`[${s.marketplace}]`)} ${installed}`,
          `    ${s.metadata.description}`,
          copy.installHint(s.source),
        ].join('\n')
      }).join('\n\n')
    })
    return
  }

  const cwd = process.cwd()
  const data = await client.searchSkills(query, { cwd, workspaceRoot: cwd })
  output(data ?? [], (skills) => {
    if (!skills.length) return copy.noMatchingSkills
    return skills.map((s) => `  ${s.name} — ${s.description}`).join('\n')
  })
}
