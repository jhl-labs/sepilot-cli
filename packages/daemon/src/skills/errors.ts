import type { SkillValidationResult } from './validator.js'

export class SkillValidationError extends Error {
  constructor(public result: SkillValidationResult) {
    super(
      `Skill validation failed: ${result.errors.join('; ') || result.warnings.join('; ')}`,
    )
    this.name = 'SkillValidationError'
  }
}

export class SkillFetchError extends Error {
  constructor(message: string, public cause?: unknown) {
    super(message)
    this.name = 'SkillFetchError'
  }
}

export class SkillPathTraversalError extends Error {
  constructor(public attemptedPath: string) {
    super(`path traversal rejected: ${attemptedPath}`)
    this.name = 'SkillPathTraversalError'
  }
}

export class SkillAlreadyExistsError extends Error {
  constructor(public skillId: string) {
    super(`skill already exists: ${skillId}`)
    this.name = 'SkillAlreadyExistsError'
  }
}

export class SkillDigestMismatchError extends Error {
  constructor(
    public expectedDigest: string,
    public actualDigest: string,
  ) {
    super(`skill source changed between preview and install: expected ${expectedDigest}, got ${actualDigest}`)
    this.name = 'SkillDigestMismatchError'
  }
}

export class SkillDigestRequiredError extends Error {
  constructor(public actualDigest: string) {
    super(`skill install requires approving the preview digest: ${actualDigest}`)
    this.name = 'SkillDigestRequiredError'
  }
}

export class SkillSourceUrlNotAllowedError extends Error {
  constructor(
    public url: string,
    public reason: string,
  ) {
    super(`skill source URL is not allowed: ${url} (${reason})`)
    this.name = 'SkillSourceUrlNotAllowedError'
  }
}
