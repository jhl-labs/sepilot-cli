import { readdir, readFile, rm, mkdir, writeFile, lstat, realpath } from 'node:fs/promises'
import { basename, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { stringify as stringifyToml } from 'smol-toml'
import type { AutonomyLevel, ISkillRegistry, SkillMetadata } from '@sepilotd/core'
import type { PolicyEngine } from '../security/policy-engine.js'
import type { ToolRegistry } from '../tools/registry.js'
import { projectCompatibleSkillRoots } from './compat.js'
import { parseSkillMd } from './loader.js'
import { validateSkill } from './validator.js'
import { SkillValidationError } from './errors.js'

export interface ValidatorDeps {
  toolRegistry: ToolRegistry
  policyEngine: PolicyEngine
  autonomy: () => AutonomyLevel
}

export interface RegisterOptions {
  force?: boolean
}

export interface FileSkillRegistryOptions {
  additionalRoots?: string[]
  /**
   * When false (default) auto-discovered project/home skills are still loaded
   * but flagged `autoDiscovered` so the `skill` tool surfaces an untrusted
   * provenance banner. When true they are treated as trusted (no banner) — an
   * explicit operator opt-in for a trusted workspace.
   */
  trustProjectSkills?: boolean
}

interface LoadRootOptions {
  trusted: boolean
}

function assertSafeSkillId(id: string): void {
  if (!id || id === '.' || id === '..' || /[/\\\0]/.test(id) || /^\.+$/.test(id)) {
    throw new Error(`unsafe skill id: ${JSON.stringify(id)}`)
  }
}

function containedSkillDir(skillsDir: string, id: string): string {
  assertSafeSkillId(id)
  const root = resolve(skillsDir)
  const target = resolve(join(root, id))
  if (target !== root && !target.startsWith(root + sep)) {
    throw new Error(`skill id escapes skills dir: ${id}`)
  }
  return target
}

export class FileSkillRegistry implements ISkillRegistry {
  private skillsDir: string
  private skills = new Map<string, { metadata: SkillMetadata; content: string }>()
  private validator: ValidatorDeps | null = null
  private additionalRoots: string[]
  private trustProjectSkills: boolean

  constructor(skillsDir: string, options: FileSkillRegistryOptions = {}) {
    this.skillsDir = skillsDir
    this.additionalRoots = options.additionalRoots ?? []
    this.trustProjectSkills = options.trustProjectSkills ?? false
  }

  setValidator(deps: ValidatorDeps): void {
    this.validator = deps
  }

  async init(): Promise<void> {
    await mkdir(this.skillsDir, { recursive: true })
    this.skills.clear()
    for (const root of this.additionalRoots) {
      await this.loadSkillRoot(root, this.skills, { trusted: this.trustProjectSkills })
    }
    await this.loadSkillRoot(this.skillsDir, this.skills, { trusted: true })
  }

  async register(skill: SkillMetadata, content: string, opts: RegisterOptions = {}): Promise<void> {
    const dir = containedSkillDir(this.skillsDir, skill.id)
    if (this.validator && !opts.force) {
      const result = validateSkill(
        skill,
        content,
        this.validator.toolRegistry,
        this.validator.policyEngine,
        this.validator.autonomy(),
      )
      if (result.errors.length > 0 || result.warnings.length > 0) {
        throw new SkillValidationError(result)
      }
    }

    this.skills.set(skill.id, { metadata: skill, content })
    await mkdir(dir, { recursive: true })
    const frontmatter: Record<string, unknown> = {
      name: skill.name,
      version: skill.version,
      description: skill.description,
      tools: skill.tools,
    }
    if (skill.author) frontmatter.author = skill.author
    if (skill.tags?.length) frontmatter.tags = skill.tags
    if (skill.autonomy_required) frontmatter.autonomy_required = skill.autonomy_required
    if (skill.source) {
      frontmatter.source_type = skill.source.type
      frontmatter.source_ref = skill.source.ref
    }
    if (skill.provenance) frontmatter.provenance = skill.provenance
    if (skill.risk_tier) frontmatter.risk_tier = skill.risk_tier
    if (skill.permissions) frontmatter.permissions = skill.permissions
    if (skill.execution) frontmatter.execution = skill.execution
    if (skill.enabled !== undefined) frontmatter.enabled = skill.enabled
    const toml = stringifyToml(frontmatter).trimEnd()
    await writeFile(join(dir, 'SKILL.md'), `+++\n${toml}\n+++\n\n${content}`, 'utf-8')
  }

  async get(id: string): Promise<{ metadata: SkillMetadata; content: string } | null> {
    return this.skills.get(id) ?? null
  }

  async getForCwd(
    id: string,
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<{ metadata: SkillMetadata; content: string } | null> {
    if (!cwd) return this.get(id)
    const skills = await this.skillsForCwd(cwd, boundaryRoot)
    return skills.get(id) ?? null
  }

  private isEnabled(m: SkillMetadata): boolean {
    return m.enabled !== false
  }

  async list(): Promise<SkillMetadata[]> {
    return Array.from(this.skills.values())
      .map(s => s.metadata)
      .filter(m => this.isEnabled(m))
  }

  async listAll(): Promise<SkillMetadata[]> {
    return Array.from(this.skills.values()).map(s => s.metadata)
  }

  async listForCwd(cwd: string | undefined, boundaryRoot?: string): Promise<SkillMetadata[]> {
    if (!cwd) return this.list()
    return Array.from((await this.skillsForCwd(cwd, boundaryRoot)).values())
      .map(s => s.metadata)
      .filter(m => this.isEnabled(m))
  }

  async listAllForCwd(cwd: string | undefined, boundaryRoot?: string): Promise<SkillMetadata[]> {
    if (!cwd) return this.listAll()
    return Array.from((await this.skillsForCwd(cwd, boundaryRoot)).values())
      .map(s => s.metadata)
  }

  async setEnabled(id: string, enabled: boolean): Promise<void> {
    const entry = this.skills.get(id)
    if (!entry) return
    await this.register({ ...entry.metadata, enabled }, entry.content, { force: true })
  }

  async remove(id: string): Promise<void> {
    const dir = containedSkillDir(this.skillsDir, id)
    this.skills.delete(id)
    try {
      await rm(dir, { recursive: true, force: true })
    } catch {
      // Ignore removal errors
    }
  }

  async search(query: string): Promise<SkillMetadata[]> {
    const q = query.toLowerCase()
    return Array.from(this.skills.values())
      .filter(s =>
        this.isEnabled(s.metadata) && (
          s.metadata.name.toLowerCase().includes(q) ||
          s.metadata.description.toLowerCase().includes(q) ||
          s.metadata.tags?.some(t => t.toLowerCase().includes(q))
        )
      )
      .map(s => s.metadata)
  }

  async searchForCwd(
    query: string,
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<SkillMetadata[]> {
    const q = query.toLowerCase()
    const source = cwd
      ? await this.skillsForCwd(cwd, boundaryRoot)
      : this.skills
    return Array.from(source.values())
      .filter(s =>
        this.isEnabled(s.metadata) && (
          s.metadata.name.toLowerCase().includes(q)
          || s.metadata.description.toLowerCase().includes(q)
          || s.metadata.tags?.some(t => t.toLowerCase().includes(q))
        )
      )
      .map(s => s.metadata)
  }

  private async skillsForCwd(
    cwd: string,
    boundaryRoot?: string,
  ): Promise<Map<string, { metadata: SkillMetadata; content: string }>> {
    // `this.skills` contains managed skills plus user-level compatibility
    // roots discovered at daemon startup. They are agent control-plane input,
    // so they remain available inside a strict workspace. Project discovery
    // below is still canonical-path bounded and cannot walk outside the root.
    const skills = new Map(this.skills)
    for (const root of await projectCompatibleSkillRoots(cwd, boundaryRoot)) {
      if (boundaryRoot && !await canonicalPathIsWithin(boundaryRoot, root)) continue
      await this.loadSkillRoot(root, skills, { trusted: this.trustProjectSkills })
    }
    return skills
  }

  private async loadSkillRoot(
    root: string,
    target: Map<string, { metadata: SkillMetadata; content: string }>,
    opts: LoadRootOptions,
  ): Promise<void> {
    await this.loadSkillDir(root, target, opts)
    let entries
    try {
      entries = await readdir(root, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      // Do not follow symlinks: an auto-discovered root that symlinks a child
      // directory could escape the workspace into arbitrary skill content.
      if (!entry.isDirectory()) continue
      await this.loadSkillDir(join(root, entry.name), target, opts)
    }
  }

  private async loadSkillDir(
    dir: string,
    target: Map<string, { metadata: SkillMetadata; content: string }>,
    opts: LoadRootOptions,
  ): Promise<void> {
    try {
      const skillPath = join(dir, 'SKILL.md')
      // Reject a symlinked SKILL.md so an auto-discovered root cannot smuggle
      // content from outside the skill directory.
      const link = await lstat(skillPath)
      if (link.isSymbolicLink()) return
      const raw = await readFile(skillPath, 'utf-8')
      const parsed = parseSkillMd(raw, basename(dir))
      const metadata: SkillMetadata = opts.trusted
        ? parsed.metadata
        : { ...parsed.metadata, autoDiscovered: true }
      // When a validator is wired, drop disk-loaded skills that fail structural
      // validation (previously the disk path bypassed validateSkill entirely).
      // Warnings (e.g. dangerous-pattern heuristics) do not block the load; the
      // untrusted banner already flags auto-discovered content.
      if (this.validator) {
        const result = validateSkill(
          metadata,
          parsed.content,
          this.validator.toolRegistry,
          this.validator.policyEngine,
          this.validator.autonomy(),
        )
        if (result.errors.length > 0) return
      }
      target.set(metadata.id, { metadata, content: parsed.content })
    } catch {
      // Skip directories that are not valid skill packages.
    }
  }
}

async function canonicalPathIsWithin(root: string, target: string): Promise<boolean> {
  try {
    const [canonicalRoot, canonicalTarget] = await Promise.all([
      realpath(root),
      realpath(target),
    ])
    const rel = relative(canonicalRoot, canonicalTarget)
    return rel === '' || (
      rel !== '..'
      && !rel.startsWith(`..${sep}`)
      && !isAbsolute(rel)
    )
  } catch {
    return false
  }
}
