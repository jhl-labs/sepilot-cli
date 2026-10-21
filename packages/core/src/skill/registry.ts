import type { SkillMetadata } from './types.js'

export interface ISkillRegistry {
  register(skill: SkillMetadata, content: string): Promise<void>
  get(id: string): Promise<{ metadata: SkillMetadata; content: string } | null>
  list(): Promise<SkillMetadata[]>
  remove(id: string): Promise<void>
  search(query: string): Promise<SkillMetadata[]>
  listAll(): Promise<SkillMetadata[]>
  setEnabled(id: string, enabled: boolean): Promise<void>
  getForCwd?(
    id: string,
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<{ metadata: SkillMetadata; content: string } | null>
  listForCwd?(cwd: string | undefined, boundaryRoot?: string): Promise<SkillMetadata[]>
  listAllForCwd?(cwd: string | undefined, boundaryRoot?: string): Promise<SkillMetadata[]>
  searchForCwd?(
    query: string,
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<SkillMetadata[]>
}
