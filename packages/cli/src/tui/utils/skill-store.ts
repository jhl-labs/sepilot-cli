import type { DaemonSkill, MarketplaceSkillSearchResult } from '@sepilotd/api-client'

const INSTALLED_SKILL_DESCRIPTION_MAX = 96
const INSTALLED_SKILL_LABEL_MAX = 4

type SkillCategory = {
  id: string
  title: string
  tags: string[]
}

type SkillSourceGroup = {
  id: string
  title: string
  note: string
}

type SkillCategoryBucket = {
  category: SkillCategory
  skills: DaemonSkill[]
}

type SkillSourceBucket = {
  group: SkillSourceGroup
  categories: Map<string, SkillCategoryBucket>
  count: number
}

const SKILL_CATEGORIES: SkillCategory[] = [
  {
    id: 'development',
    title: 'Development',
    tags: [
      'acp',
      'architecture',
      'authoring',
      'bisect',
      'bug',
      'build',
      'code-review',
      'coverage',
      'debug',
      'dependencies',
      'external-agent',
      'integration',
      'pr',
      'quality',
      'refactor',
      'regression',
      'review',
      'safe-transform',
      'skills',
      'tdd',
      'test',
      'tests',
      'upgrade',
    ],
  },
  {
    id: 'operations',
    title: 'Operations',
    tags: [
      'certificates',
      'ci',
      'cluster-maintenance',
      'cve',
      'docker',
      'enterprise',
      'github',
      'iac',
      'kubernetes',
      'kubectl',
      'network-setup',
      'ops',
      'owasp',
      'proxy',
      'release',
      'sandbox',
      'security',
      'semver',
      'sre',
      'workflows',
    ],
  },
  {
    id: 'connectors',
    title: 'Connectors & automation',
    tags: [
      'atlassian',
      'automation',
      'calendar',
      'confluence',
      'connector',
      'connectors',
      'email',
      'gmail',
      'imap',
      'jira',
      'mcp',
      'notion',
      'recipes',
      'slack',
      'workspace',
      'zapier',
    ],
  },
  {
    id: 'research',
    title: 'Research & data',
    tags: [
      'analysis',
      'browser',
      'citations',
      'csv',
      'data',
      'deep-research',
      'duckdb',
      'fact-check',
      'literature-review',
      'pandas',
      'parquet',
      'playwright',
      'research',
      'scraping',
      'screenshot',
      'web',
    ],
  },
  {
    id: 'personal',
    title: 'Personal productivity',
    tags: [
      'agenda',
      'briefing',
      'correspondence',
      'daily',
      'decisions',
      'digest',
      'inbox',
      'meeting',
      'morning',
      'notes',
      'personal',
      'reading',
      'reflection',
      'reminders',
      'tasks',
      'todo',
    ],
  },
  {
    id: 'collaboration',
    title: 'Work collaboration',
    tags: [
      'accountability',
      'action-items',
      'community',
      'discussions',
      'governance',
      'issues',
      'organisational-memory',
      'projects',
      'q-and-a',
      'sprint',
      'tickets',
      'tracking',
      'triage',
      'work',
    ],
  },
  {
    id: 'writing',
    title: 'Writing & docs',
    tags: [
      'api-docs',
      'changelog',
      'docs',
      'documentation',
      'executive-summary',
      'github-pages',
      'incident',
      'korean',
      'polish',
      'publishing',
      'readme',
      'report',
      'static-site',
      'status',
      'technical-blog',
      'tone',
      'writing',
    ],
  },
]

const GENERAL_SKILL_CATEGORY: SkillCategory = {
  id: 'general',
  title: 'General',
  tags: [],
}

const SOURCE_GROUPS: SkillSourceGroup[] = [
  {
    id: 'builtin-active',
    title: 'Built-in active',
    note: 'Bundled skills already visible to the agent.',
  },
  {
    id: 'builtin-opt-in',
    title: 'Built-in opt-in',
    note: 'Bundled skills disabled until explicitly enabled.',
  },
  {
    id: 'source-installed',
    title: 'Installed from sources',
    note: 'Skills installed from a marketplace, git repo, or URL.',
  },
  {
    id: 'local-custom',
    title: 'Local / custom',
    note: 'User or project skills without a known remote source.',
  },
]

export const SKILLS_COMMAND_USAGE = [
  'Usage:',
  '  /skills                         Open the installed skill manager',
  '  /skills installed [query]       Open installed skill manager (incl. disabled)',
  '  /skills manage [query]          Alias for /skills installed',
  '  /skills enable <id>             Enable a skill',
  '  /skills disable <id>            Disable a skill',
  '  /skills store [query]           Open the interactive skill catalog',
  '  /skills search [query]          Search configured skill sources in the picker',
  '  /skills install <source>        Install a skill from a catalog result, URL, or path',
  '  /run <id> [prompt]              Run an installed skill; /run and @skill: support picker selection',
].join('\n')

function compactText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

function truncateText(value: string, maxLength: number): string {
  const text = compactText(value)
  if (text.length <= maxLength) return text
  return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
}

function formatSkillDescription(skill: Pick<DaemonSkill, 'description'>): string | null {
  if (!skill.description) return null
  return truncateText(skill.description, INSTALLED_SKILL_DESCRIPTION_MAX)
}

function formatSkillLabels(skill: Pick<DaemonSkill, 'tools' | 'tags'>): string | null {
  const labels = [
    ...(skill.tools ?? []).map((tool) => `tool:${tool}`),
    ...(skill.tags ?? []).map((tag) => `#${tag}`),
  ]
  if (labels.length === 0) return null
  const visible = labels.slice(0, INSTALLED_SKILL_LABEL_MAX)
  const suffix = labels.length > visible.length ? `, +${labels.length - visible.length} more` : ''
  return `${visible.join(', ')}${suffix}`
}

function normalizedSkillTags(skill: Pick<DaemonSkill, 'tags'>): Set<string> {
  return new Set((skill.tags ?? []).map((tag) => tag.toLowerCase()))
}

function getSkillCategory(skill: Pick<DaemonSkill, 'tags'>): SkillCategory {
  const tags = normalizedSkillTags(skill)
  return (
    SKILL_CATEGORIES.find((category) => category.tags.some((tag) => tags.has(tag))) ??
    GENERAL_SKILL_CATEGORY
  )
}

export function getSkillCategoryTitle(skill: Pick<DaemonSkill, 'tags'>): string {
  return getSkillCategory(skill).title
}

function getSkillSourceGroup(
  skill: Pick<DaemonSkill, 'enabled' | 'source' | 'tags'>,
): SkillSourceGroup {
  const tags = normalizedSkillTags(skill)
  const builtin = tags.has('builtin')
  if (builtin && skill.enabled === false) {
    return SOURCE_GROUPS[1]!
  }
  if (builtin) {
    return SOURCE_GROUPS[0]!
  }
  if (skill.source) {
    return SOURCE_GROUPS[2]!
  }
  return SOURCE_GROUPS[3]!
}

function formatSkillSource(skill: Pick<DaemonSkill, 'source'>): string | null {
  if (!skill.source) return null
  return `${skill.source.type}:${skill.source.ref}`
}

function countEnabled(skills: DaemonSkill[]): number {
  return skills.filter((skill) => skill.enabled !== false).length
}

function sourceGroupSortIndex(id: string): number {
  const index = SOURCE_GROUPS.findIndex((group) => group.id === id)
  return index === -1 ? SOURCE_GROUPS.length : index
}

function categorySortIndex(id: string): number {
  if (id === GENERAL_SKILL_CATEGORY.id) return SKILL_CATEGORIES.length
  const index = SKILL_CATEGORIES.findIndex((category) => category.id === id)
  return index === -1 ? SKILL_CATEGORIES.length + 1 : index
}

function compareSkillNames(a: DaemonSkill, b: DaemonSkill): number {
  return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
}

function formatInstalledSkillRow(skill: DaemonSkill): string {
  const state = skill.enabled === false ? '✗ disabled' : '✓ enabled'
  const description = formatSkillDescription(skill)
  const labels = formatSkillLabels(skill)
  const source = formatSkillSource(skill)
  return [
    `- ${state}  ${skill.name} v${skill.version}`,
    `  id: ${skill.id}`,
    source ? `  source: ${source}` : null,
    description ? `  ${description}` : null,
    labels ? `  labels: ${labels}` : null,
  ]
    .filter(Boolean)
    .join('\n')
}

export function formatInstalledSkills(skills: DaemonSkill[]): string {
  if (skills.length === 0) {
    return `No installed skills found.\n\n${SKILLS_COMMAND_USAGE}`
  }

  const sourceGroups = new Map<string, SkillSourceBucket>()

  for (const skill of skills) {
    const group = getSkillSourceGroup(skill)
    const category = getSkillCategory(skill)
    let sourceBucket = sourceGroups.get(group.id)
    if (!sourceBucket) {
      sourceBucket = { group, categories: new Map(), count: 0 }
      sourceGroups.set(group.id, sourceBucket)
    }
    sourceBucket.count += 1
    let categoryBucket = sourceBucket.categories.get(category.id)
    if (!categoryBucket) {
      categoryBucket = { category, skills: [] }
      sourceBucket.categories.set(category.id, categoryBucket)
    }
    categoryBucket.skills.push(skill)
  }

  const activeCount = countEnabled(skills)
  const optInCount = skills.length - activeCount
  const renderedGroups = Array.from(sourceGroups.values())
    .sort((a, b) => sourceGroupSortIndex(a.group.id) - sourceGroupSortIndex(b.group.id))
    .flatMap(({ group, categories, count }) => {
      const categoryBlocks = Array.from(categories.values())
        .sort((a, b) => categorySortIndex(a.category.id) - categorySortIndex(b.category.id))
        .map(({ category, skills: categorySkills }) =>
          [
            `${category.title} (${categorySkills.length}):`,
            categorySkills
              .slice()
              .sort(compareSkillNames)
              .map(formatInstalledSkillRow)
              .join('\n\n'),
          ].join('\n'),
        )

      return [`${group.title} (${count}):`, `  note: ${group.note}`, ...categoryBlocks].join('\n')
    })

  return [
    `Installed skills (${skills.length}):`,
    `Active ${activeCount} · Opt-in/disabled ${optInCount}`,
    '',
    renderedGroups.join('\n\n'),
    '',
    'Open /skills for checkbox toggles · /skills store to browse sources.',
  ].join('\n')
}

export function formatMarketplaceSkills(
  results: MarketplaceSkillSearchResult[],
  query: string,
): string {
  if (results.length === 0) {
    return `No configured skill sources matched "${query}".`
  }

  return [
    `Skill catalog results for "${query}" (${results.length}):`,
    ...results.map((result, index) => {
      const metadata = result.metadata
      const status = result.installed ? 'installed' : 'not installed'
      return [
        `${index + 1}. ${metadata.name} v${metadata.version}`,
        metadata.description ? `   ${metadata.description}` : null,
        `   category: ${getSkillCategoryTitle(metadata)}`,
        `   source catalog: ${result.marketplace} (${status})`,
        `   source: ${result.source}`,
        result.installed ? null : `   install: /skills install ${result.source}`,
      ]
        .filter(Boolean)
        .join('\n')
    }),
  ].join('\n')
}

export function formatSkillInstallResult(skills: DaemonSkill[]): string {
  if (skills.length === 0) {
    return 'Skill install completed, but the daemon did not return any installed skills.'
  }

  return [
    `Installed ${skills.length} skill${skills.length === 1 ? '' : 's'}:`,
    ...skills.map((skill) => `- ${skill.name} v${skill.version} (${skill.id})`),
  ].join('\n')
}
