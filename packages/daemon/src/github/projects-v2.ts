export type GitHubProjectOwnerType = 'user' | 'organization'

export interface GitHubProjectRef {
  ownerType: GitHubProjectOwnerType
  owner: string
  number: number
}

export interface GitHubProjectSummary extends GitHubProjectRef {
  id: string
  name: string
  url: string
  updatedAt: string
  viewerCanUpdate: boolean
  ownerName?: string
}

export interface GitHubProjectsList {
  provider: 'github-projects-v2'
  login: string
  projects: GitHubProjectSummary[]
  pulledAt: number
}

export interface GitHubProjectColumn {
  id: string
  name: string
  position: number
  remoteOptionId: string | null
}

export interface GitHubProjectCard {
  id: string
  columnId: string
  title: string
  body?: string
  url?: string
  assignees: string[]
  createdAt: string
  updatedAt: string
  fields: {
    contentId?: string
    contentType?: string
    draftIssueId?: string
    repository?: string
    number?: number
    status?: string
    statusOptionId?: string
  }
}

export interface GitHubProjectBoard {
  provider: 'github-projects-v2'
  id: string
  owner: string
  ownerType: GitHubProjectOwnerType
  number: number
  name: string
  url: string
  updatedAt: string
  viewerCanUpdate: boolean
  statusFieldId: string
  columns: GitHubProjectColumn[]
  cards: GitHubProjectCard[]
  pulledAt: number
}

export interface GitHubCreateProjectCardInput {
  title: string
  body?: string
  columnId?: string
}

export interface GitHubUpdateProjectDraftInput {
  draftIssueId: string
  title?: string
  body?: string
}

interface GraphQlResponse<T> {
  data?: T
  errors?: Array<{ message?: string }>
}

interface ProjectSettings {
  id: string
  statusFieldId: string
  columns: GitHubProjectColumn[]
}

const GITHUB_GRAPHQL_URL = 'https://api.github.com/graphql'
const NO_STATUS_COLUMN_ID = '__github_no_status__'
const MAX_PROJECT_ITEMS = 500
const MAX_PROJECT_LIST_OWNERS = 50
const MAX_PROJECT_LIST_PER_OWNER = 25

function projectOwnerField(ownerType: GitHubProjectOwnerType): 'user' | 'organization' {
  return ownerType === 'organization' ? 'organization' : 'user'
}

function githubHeaders(token: string): HeadersInit {
  return {
    accept: 'application/vnd.github+json',
    authorization: `Bearer ${token}`,
    'content-type': 'application/json',
    'user-agent': 'sepilotd',
    'x-github-api-version': '2022-11-28',
  }
}

async function githubGraphql<T>(
  token: string,
  query: string,
  variables: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(GITHUB_GRAPHQL_URL, {
    method: 'POST',
    headers: githubHeaders(token),
    body: JSON.stringify({ query, variables }),
  })
  const text = await response.text()
  let payload: GraphQlResponse<T>
  try {
    payload = JSON.parse(text) as GraphQlResponse<T>
  } catch {
    throw new Error(`GitHub GraphQL returned invalid JSON (${response.status}).`)
  }
  if (!response.ok) {
    const message = payload.errors?.map((error) => error.message).filter(Boolean).join('; ')
    throw new Error(`GitHub GraphQL failed (${response.status}): ${message || response.statusText}`)
  }
  if (payload.errors?.length) {
    const message = payload.errors.map((error) => error.message || 'Unknown GitHub error').join('; ')
    throw new Error(`GitHub GraphQL error: ${message}`)
  }
  if (!payload.data) {
    throw new Error('GitHub GraphQL response did not include data.')
  }
  return payload.data
}

function projectSelection(ownerType: GitHubProjectOwnerType): string {
  const ownerField = projectOwnerField(ownerType)
  return `
    owner: ${ownerField}(login: $login) {
      projectV2(number: $number) {
        id
        title
        url
        updatedAt
        viewerCanUpdate
        field(name: "Status") {
          __typename
          ... on ProjectV2SingleSelectField {
            id
            name
            options {
              id
              name
            }
          }
        }
        items(first: 100, after: $after) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            id
            createdAt
            updatedAt
            content {
              __typename
              ... on DraftIssue {
                id
                title
                body
                createdAt
                updatedAt
                assignees(first: 10) {
                  nodes { login }
                }
              }
              ... on Issue {
                id
                title
                body
                url
                number
                createdAt
                updatedAt
                repository { nameWithOwner }
                assignees(first: 10) {
                  nodes { login }
                }
              }
              ... on PullRequest {
                id
                title
                body
                url
                number
                createdAt
                updatedAt
                repository { nameWithOwner }
                assignees(first: 10) {
                  nodes { login }
                }
              }
            }
            fieldValues(first: 30) {
              nodes {
                __typename
                ... on ProjectV2ItemFieldSingleSelectValue {
                  name
                  optionId
                  field {
                    ... on ProjectV2FieldCommon {
                      id
                      name
                    }
                  }
                }
              }
            }
          }
        }
      }
    }`
}

function projectSummarySelection(): string {
  return `
    id
    number
    title
    url
    updatedAt
    viewerCanUpdate`
}

function settingsSelection(ownerType: GitHubProjectOwnerType): string {
  const ownerField = projectOwnerField(ownerType)
  return `
    owner: ${ownerField}(login: $login) {
      projectV2(number: $number) {
        id
        field(name: "Status") {
          __typename
          ... on ProjectV2SingleSelectField {
            id
            options {
              id
              name
            }
          }
        }
      }
    }`
}

function mapProjectSummary(
  value: unknown,
  ownerType: GitHubProjectOwnerType,
  owner: string,
  ownerName?: string,
): GitHubProjectSummary | null {
  if (!value || typeof value !== 'object') return null
  const project = value as Record<string, unknown>
  if (
    typeof project.id !== 'string'
    || typeof project.title !== 'string'
    || typeof project.number !== 'number'
  ) {
    return null
  }
  return {
    id: project.id,
    ownerType,
    owner,
    ownerName,
    number: project.number,
    name: project.title,
    url: typeof project.url === 'string' ? project.url : '',
    updatedAt: typeof project.updatedAt === 'string' ? project.updatedAt : '',
    viewerCanUpdate: project.viewerCanUpdate === true,
  }
}

function normalizeColumns(statusField: unknown, includeNoStatus: boolean): GitHubProjectColumn[] {
  const field = statusField && typeof statusField === 'object'
    ? statusField as { __typename?: unknown; id?: unknown; options?: unknown }
    : null
  if (!field || field.__typename !== 'ProjectV2SingleSelectField' || typeof field.id !== 'string') {
    throw new Error('GitHub Project must have a single-select Status field to sync as a kanban board.')
  }
  const options = Array.isArray(field.options) ? field.options : []
  const columns = options
    .filter((option): option is { id: string; name: string } => {
      return Boolean(
        option
          && typeof option === 'object'
          && typeof (option as { id?: unknown }).id === 'string'
          && typeof (option as { name?: unknown }).name === 'string',
      )
    })
    .map((option, index) => ({
      id: option.id,
      name: option.name,
      position: includeNoStatus ? index + 1 : index,
      remoteOptionId: option.id,
    }))
  if (columns.length === 0) {
    throw new Error('GitHub Project Status field does not have any options.')
  }
  return includeNoStatus
    ? [{ id: NO_STATUS_COLUMN_ID, name: '상태 없음', position: 0, remoteOptionId: null }, ...columns]
    : columns
}

function projectFromData(data: unknown): Record<string, unknown> {
  const owner = data && typeof data === 'object' ? (data as { owner?: unknown }).owner : null
  if (!owner || typeof owner !== 'object') {
    throw new Error('GitHub Project owner was not found.')
  }
  const project = (owner as { projectV2?: unknown }).projectV2
  if (!project || typeof project !== 'object') {
    throw new Error('GitHub Project was not found or is not accessible.')
  }
  return project as Record<string, unknown>
}

function statusValueForItem(
  item: Record<string, unknown>,
  statusFieldId: string,
): { name?: string; optionId?: string } {
  const fieldValues = item.fieldValues && typeof item.fieldValues === 'object'
    ? (item.fieldValues as { nodes?: unknown }).nodes
    : []
  const values = Array.isArray(fieldValues) ? fieldValues : []
  for (const value of values) {
    if (!value || typeof value !== 'object') continue
    const record = value as {
      __typename?: unknown
      name?: unknown
      optionId?: unknown
      field?: { id?: unknown; name?: unknown }
    }
    if (record.__typename !== 'ProjectV2ItemFieldSingleSelectValue') continue
    if (record.field?.id !== statusFieldId && record.field?.name !== 'Status') continue
    return {
      name: typeof record.name === 'string' ? record.name : undefined,
      optionId: typeof record.optionId === 'string' ? record.optionId : undefined,
    }
  }
  return {}
}

function assigneeLogins(content: Record<string, unknown> | null): string[] {
  const nodes = content?.assignees && typeof content.assignees === 'object'
    ? (content.assignees as { nodes?: unknown }).nodes
    : []
  if (!Array.isArray(nodes)) return []
  return nodes
    .map((node) => node && typeof node === 'object' ? (node as { login?: unknown }).login : null)
    .filter((login): login is string => typeof login === 'string' && login.length > 0)
}

function mapItemToCard(
  item: Record<string, unknown>,
  board: { statusFieldId: string },
): GitHubProjectCard | null {
  if (typeof item.id !== 'string') return null
  const content = item.content && typeof item.content === 'object'
    ? item.content as Record<string, unknown>
    : null
  const status = statusValueForItem(item, board.statusFieldId)
  const contentType = typeof content?.__typename === 'string' ? content.__typename : undefined
  const title = typeof content?.title === 'string' && content.title.trim()
    ? content.title
    : `Project item ${item.id}`
  const updatedAt = typeof content?.updatedAt === 'string'
    ? content.updatedAt
    : typeof item.updatedAt === 'string'
      ? item.updatedAt
      : new Date().toISOString()
  const createdAt = typeof content?.createdAt === 'string'
    ? content.createdAt
    : typeof item.createdAt === 'string'
      ? item.createdAt
      : updatedAt

  return {
    id: item.id,
    columnId: status.optionId ?? NO_STATUS_COLUMN_ID,
    title,
    body: typeof content?.body === 'string' ? content.body : undefined,
    url: typeof content?.url === 'string' ? content.url : undefined,
    assignees: assigneeLogins(content),
    createdAt,
    updatedAt,
    fields: {
      contentId: typeof content?.id === 'string' ? content.id : undefined,
      contentType,
      draftIssueId: contentType === 'DraftIssue' && typeof content?.id === 'string' ? content.id : undefined,
      repository:
        content?.repository && typeof content.repository === 'object'
          ? ((content.repository as { nameWithOwner?: unknown }).nameWithOwner as string | undefined)
          : undefined,
      number: typeof content?.number === 'number' ? content.number : undefined,
      status: status.name,
      statusOptionId: status.optionId,
    },
  }
}

async function loadProjectSettings(
  token: string,
  ref: GitHubProjectRef,
): Promise<ProjectSettings> {
  const query = `
    query SepilotProjectSettings($login: String!, $number: Int!) {
      ${settingsSelection(ref.ownerType)}
    }`
  const data = await githubGraphql<Record<string, unknown>>(token, query, {
    login: ref.owner,
    number: ref.number,
  })
  const project = projectFromData(data)
  if (typeof project.id !== 'string') {
    throw new Error('GitHub Project response did not include project id.')
  }
  const field = project.field as { id?: unknown } | undefined
  if (!field || typeof field.id !== 'string') {
    throw new Error('GitHub Project must have a single-select Status field to sync as a kanban board.')
  }
  return {
    id: project.id,
    statusFieldId: field.id,
    columns: normalizeColumns(project.field, false),
  }
}

export async function listGitHubProjectsV2(
  token: string,
  input: { query?: string } = {},
): Promise<GitHubProjectsList> {
  const query = `
    query SepilotProjectPicker($projectFirst: Int!, $orgFirst: Int!, $orgProjectFirst: Int!, $query: String) {
      viewer {
        login
        projectsV2(first: $projectFirst, query: $query) {
          nodes {
            ${projectSummarySelection()}
          }
        }
        organizations(first: $orgFirst) {
          nodes {
            login
            name
            projectsV2(first: $orgProjectFirst, query: $query) {
              nodes {
                ${projectSummarySelection()}
              }
            }
          }
        }
      }
    }`
  const search = input.query?.trim() || null
  const data = await githubGraphql<{
    viewer?: {
      login?: unknown
      projectsV2?: { nodes?: unknown }
      organizations?: { nodes?: unknown }
    }
  }>(token, query, {
    projectFirst: MAX_PROJECT_LIST_PER_OWNER,
    orgFirst: MAX_PROJECT_LIST_OWNERS,
    orgProjectFirst: MAX_PROJECT_LIST_PER_OWNER,
    query: search,
  })
  const viewer = data.viewer
  if (!viewer || typeof viewer.login !== 'string') {
    throw new Error('GitHub viewer information was not returned.')
  }

  const projects: GitHubProjectSummary[] = []
  const viewerProjectNodes = Array.isArray(viewer.projectsV2?.nodes) ? viewer.projectsV2.nodes : []
  for (const node of viewerProjectNodes) {
    const project = mapProjectSummary(node, 'user', viewer.login)
    if (project) projects.push(project)
  }

  const orgNodes = Array.isArray(viewer.organizations?.nodes) ? viewer.organizations.nodes : []
  for (const orgNode of orgNodes) {
    if (!orgNode || typeof orgNode !== 'object') continue
    const org = orgNode as { login?: unknown; name?: unknown; projectsV2?: { nodes?: unknown } }
    if (typeof org.login !== 'string') continue
    const projectNodes = Array.isArray(org.projectsV2?.nodes) ? org.projectsV2.nodes : []
    for (const node of projectNodes) {
      const project = mapProjectSummary(
        node,
        'organization',
        org.login,
        typeof org.name === 'string' && org.name.trim() ? org.name : undefined,
      )
      if (project) projects.push(project)
    }
  }

  const seen = new Set<string>()
  const deduped = projects
    .filter((project) => {
      const key = `${project.ownerType}:${project.owner}:${project.number}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
    .sort((a, b) => Date.parse(b.updatedAt || '') - Date.parse(a.updatedAt || ''))

  return {
    provider: 'github-projects-v2',
    login: viewer.login,
    projects: deduped,
    pulledAt: Date.now(),
  }
}

export async function getGitHubProjectBoard(
  token: string,
  ref: GitHubProjectRef,
): Promise<GitHubProjectBoard> {
  let after: string | null = null
  let project: Record<string, unknown> | null = null
  const items: Record<string, unknown>[] = []

  do {
    const query = `
      query SepilotProjectBoard($login: String!, $number: Int!, $after: String) {
        ${projectSelection(ref.ownerType)}
      }`
    const data = await githubGraphql<Record<string, unknown>>(token, query, {
      login: ref.owner,
      number: ref.number,
      after,
    })
    project = projectFromData(data)
    const itemConnection = project.items && typeof project.items === 'object'
      ? project.items as { nodes?: unknown; pageInfo?: { hasNextPage?: unknown; endCursor?: unknown } }
      : null
    const nodes = Array.isArray(itemConnection?.nodes) ? itemConnection.nodes : []
    for (const node of nodes) {
      if (node && typeof node === 'object') items.push(node as Record<string, unknown>)
    }
    const pageInfo = itemConnection?.pageInfo
    after = pageInfo?.hasNextPage === true && typeof pageInfo.endCursor === 'string'
      ? pageInfo.endCursor
      : null
  } while (after && items.length < MAX_PROJECT_ITEMS)

  if (!project) throw new Error('GitHub Project was not found or is not accessible.')
  if (typeof project.id !== 'string' || typeof project.title !== 'string') {
    throw new Error('GitHub Project response did not include required fields.')
  }
  const statusField = project.field as { id?: unknown } | undefined
  if (!statusField || typeof statusField.id !== 'string') {
    throw new Error('GitHub Project must have a single-select Status field to sync as a kanban board.')
  }
  const mappedCards = items
    .map((item) => mapItemToCard(item, { statusFieldId: statusField.id as string }))
    .filter((card): card is GitHubProjectCard => Boolean(card))
  const includeNoStatus = mappedCards.some((card) => card.columnId === NO_STATUS_COLUMN_ID)

  return {
    provider: 'github-projects-v2',
    id: project.id,
    owner: ref.owner,
    ownerType: ref.ownerType,
    number: ref.number,
    name: project.title,
    url: typeof project.url === 'string' ? project.url : '',
    updatedAt: typeof project.updatedAt === 'string' ? project.updatedAt : new Date().toISOString(),
    viewerCanUpdate: project.viewerCanUpdate === true,
    statusFieldId: statusField.id,
    columns: normalizeColumns(project.field, includeNoStatus),
    cards: mappedCards,
    pulledAt: Date.now(),
  }
}

export async function createGitHubProjectDraftCard(
  token: string,
  ref: GitHubProjectRef,
  input: GitHubCreateProjectCardInput,
): Promise<{ itemId: string }> {
  const title = input.title.trim()
  if (!title) throw new Error('Card title is required.')
  const settings = await loadProjectSettings(token, ref)
  const addDraftMutation = `
    mutation SepilotAddProjectDraft($input: AddProjectV2DraftIssueInput!) {
      addProjectV2DraftIssue(input: $input) {
        projectItem { id }
      }
    }`
  const draftData = await githubGraphql<{
    addProjectV2DraftIssue?: { projectItem?: { id?: unknown } }
  }>(token, addDraftMutation, {
    input: {
      projectId: settings.id,
      title,
      body: input.body ?? '',
    },
  })
  const itemId = draftData.addProjectV2DraftIssue?.projectItem?.id
  if (typeof itemId !== 'string') {
    throw new Error('GitHub did not return the created project item id.')
  }
  if (input.columnId && input.columnId !== NO_STATUS_COLUMN_ID) {
    await moveGitHubProjectCard(token, ref, { itemId, columnId: input.columnId })
  }
  return { itemId }
}

export async function moveGitHubProjectCard(
  token: string,
  ref: GitHubProjectRef,
  input: { itemId: string; columnId: string },
): Promise<{ ok: true; itemId: string; columnId: string; updatedAt: number }> {
  if (!input.itemId.trim()) throw new Error('Project item id is required.')
  if (!input.columnId.trim() || input.columnId === NO_STATUS_COLUMN_ID) {
    throw new Error('A GitHub Status option id is required to move a project item.')
  }
  const settings = await loadProjectSettings(token, ref)
  if (!settings.columns.some((column) => column.id === input.columnId)) {
    throw new Error('Target column is not a Status option on this GitHub Project.')
  }
  const mutation = `
    mutation SepilotMoveProjectCard($input: UpdateProjectV2ItemFieldValueInput!) {
      updateProjectV2ItemFieldValue(input: $input) {
        projectV2Item { id }
      }
    }`
  await githubGraphql(token, mutation, {
    input: {
      projectId: settings.id,
      itemId: input.itemId,
      fieldId: settings.statusFieldId,
      value: {
        singleSelectOptionId: input.columnId,
      },
    },
  })
  return { ok: true, itemId: input.itemId, columnId: input.columnId, updatedAt: Date.now() }
}

export async function updateGitHubProjectDraftCard(
  token: string,
  input: GitHubUpdateProjectDraftInput,
): Promise<{ ok: true; draftIssueId: string; updatedAt: number }> {
  const draftIssueId = input.draftIssueId.trim()
  if (!draftIssueId) throw new Error('Draft issue id is required.')
  if (input.title != null && !input.title.trim()) throw new Error('Draft issue title cannot be empty.')
  const mutation = `
    mutation SepilotUpdateProjectDraft($input: UpdateProjectV2DraftIssueInput!) {
      updateProjectV2DraftIssue(input: $input) {
        draftIssue { id }
      }
    }`
  await githubGraphql(token, mutation, {
    input: {
      draftIssueId,
      ...(input.title != null ? { title: input.title.trim() } : {}),
      ...(input.body != null ? { body: input.body } : {}),
    },
  })
  return { ok: true, draftIssueId, updatedAt: Date.now() }
}
