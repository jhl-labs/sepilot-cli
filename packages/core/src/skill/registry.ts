import type { SkillMetadata } from './types.js'

export interface ISkillRegistry {
  register(skill: SkillMetadata, content: string): Promise<void>
  get(id: string): Promise<{ metadata: SkillMetadata; content: string } | null>
  list(): Promise<SkillMetadata[]>
  remove(id: string): Promise<void>
  search(query: string): Promise<SkillMetadata[]>
  /** List every skill, including ones with `enabled: false`. */
  listAll(): Promise<SkillMetadata[]>
  /** Flip a skill's `enabled` flag and persist it. No-op for unknown ids. */
  setEnabled(id: string, enabled: boolean): Promise<void>
  /** Resolve project-compatible skills for one request working directory. */
  getForCwd?(
    id: string,
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<{ metadata: SkillMetadata; content: string } | null>
  /** List enabled managed, global-compatible, and project-local skills. */
  listForCwd?(cwd: string | undefined, boundaryRoot?: string): Promise<SkillMetadata[]>
  /** CWD-aware management list including disabled skills. */
  listAllForCwd?(cwd: string | undefined, boundaryRoot?: string): Promise<SkillMetadata[]>
  /** Search managed, global-compatible, and project-local skills. */
  searchForCwd?(
    query: string,
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<SkillMetadata[]>
}
