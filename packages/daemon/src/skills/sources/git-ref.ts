import type { SimpleGit } from 'simple-git'
import { SkillFetchError } from '../errors.js'

const GIT_OBJECT_ID_PATTERN = /^[0-9a-f]{7,64}$/i

export function gitDetachedSwitchArgs(ref: string): string[] {
  const trimmed = ref.trim()
  if (!trimmed) {
    throw new SkillFetchError('Git ref is empty')
  }
  return ['switch', '--detach', '--', trimmed]
}

export async function checkoutDetachedGitRef(
  git: Pick<SimpleGit, 'raw'>,
  ref: string,
): Promise<void> {
  await git.raw(gitDetachedSwitchArgs(ref))
}

export function validateGitObjectId(
  value: string,
  label = 'Git object id',
): string {
  const trimmed = value.trim()
  if (!GIT_OBJECT_ID_PATTERN.test(trimmed)) {
    throw new SkillFetchError(`${label} must be a hex object id`)
  }
  return trimmed
}
