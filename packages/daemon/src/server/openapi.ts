export type OpenApiMethod = 'get' | 'post' | 'put' | 'delete' | 'patch' | 'options'

export interface DiscoveredOpenApiRoute {
  method: OpenApiMethod
  path: string
  websocket?: boolean
}

interface RouteDiscoveryInput {
  method: string | string[]
  url: string
  websocket?: boolean
}

export type OpenApiSchema = Record<string, unknown>

export interface OpenApiParameter {
  name: string
  in: 'path' | 'query' | 'header' | 'cookie'
  required?: boolean
  schema?: OpenApiSchema
}

export interface OpenApiParameterReference {
  $ref: string
}

export type OpenApiOperationParameter = OpenApiParameter | OpenApiParameterReference
export type OpenApiComponentSchemaMap = Record<string, OpenApiSchema>
export type OpenApiComponentParameterMap = Record<string, OpenApiParameter>

export interface OpenApiComponentOverrides {
  schemas?: OpenApiComponentSchemaMap
  parameters?: OpenApiComponentParameterMap
}

interface OpenApiOperation {
  summary?: string
  tags?: string[]
  description?: string
  parameters?: OpenApiOperationParameter[]
  requestBody?: OpenApiSchema
  responses?: Record<string, unknown>
  security?: Array<Record<string, string[]>>
}

type OperationOverrideMap = Partial<Record<OpenApiMethod, OpenApiOperation>>
export type OpenApiOverrideMap = Record<string, OperationOverrideMap>

const METHOD_ORDER: OpenApiMethod[] = ['get', 'post', 'put', 'delete', 'patch', 'options']

const TAG_BY_SEGMENT: Record<string, string> = {
  agents: 'Agents',
  auth: 'Auth',
  chat: 'Chat',
  config: 'Config',
  cron: 'Cron',
  devices: 'Devices',
  estimate: 'Chat',
  files: 'Files',
  health: 'System',
  memory: 'Memory',
  metrics: 'System',
  'openapi.json': 'System',
  personas: 'Personas',
  plugins: 'Plugins',
  projects: 'Projects',
  sessions: 'Sessions',
  'skill-store': 'Skill Store',
  skills: 'Skills',
  usage: 'Usage',
  webhooks: 'Webhooks',
  ws: 'Chat',
}

const OPERATION_OVERRIDES: OpenApiOverrideMap = {
  '/api/v1/openapi.json': {
    get: {
      summary: 'OpenAPI specification',
      tags: ['System'],
      responses: { 200: { description: 'OpenAPI document' } },
    },
  },
  '/api/v1/ws': {
    get: {
      summary: 'WebSocket connection',
      tags: ['Chat'],
      description: 'Upgrade to WebSocket for real-time streaming. Browser surfaces should prefer /api/v1/chat/stream.',
      responses: { 101: { description: 'Switching protocols' } },
    },
  },
}

const COMPONENT_OVERRIDES: OpenApiComponentOverrides = {
  schemas: {},
}

function normalizePath(path: string): string {
  const trimmed = path === '/' ? path : path.replace(/\/+$/, '')
  return trimmed
    .replace(/\/\*$/g, '/{path}')
    .replace(/:([A-Za-z0-9_]+)/g, '{$1}')
}

function normalizeMethod(method: string): OpenApiMethod | null {
  const normalized = method.toLowerCase()
  if (normalized === 'head') return null
  if (METHOD_ORDER.includes(normalized as OpenApiMethod)) {
    return normalized as OpenApiMethod
  }
  return null
}

function inferPathParameters(path: string): OpenApiParameter[] {
  return Array.from(path.matchAll(/\{([^}]+)\}/g)).map((match) => ({
    name: match[1]!,
    in: 'path',
    required: true,
    schema: { type: 'string' },
  }))
}

function isOpenApiParameterReference(
  value: OpenApiOperationParameter,
): value is OpenApiParameterReference {
  return '$ref' in value
}

function mergeParameters(
  inferred: OpenApiParameter[] | undefined,
  override: OpenApiOperationParameter[] | undefined,
): OpenApiOperationParameter[] | undefined {
  const merged: OpenApiOperationParameter[] = [...(inferred ?? [])]

  for (const parameter of override ?? []) {
    if (isOpenApiParameterReference(parameter)) {
      merged.push(parameter)
      continue
    }

    const existingIndex = merged.findIndex(
      (candidate) =>
        !isOpenApiParameterReference(candidate)
        && candidate.name === parameter.name
        && candidate.in === parameter.in,
    )

    if (existingIndex >= 0) {
      merged[existingIndex] = parameter
    } else {
      merged.push(parameter)
    }
  }

  return merged.length > 0 ? merged : undefined
}

function tagForPath(path: string): string {
  const segments = path.split('/').filter(Boolean)
  const segment = segments[2] ?? segments[segments.length - 1] ?? 'system'
  return TAG_BY_SEGMENT[segment]
    ?? segment.replace(/(^|-)(\w)/g, (_match, _sep, char: string) => ` ${char.toUpperCase()}`).trim()
}

function buildDefaultOperation(route: DiscoveredOpenApiRoute): OpenApiOperation {
  const parameters = inferPathParameters(route.path)

  const operation: OpenApiOperation = {
    summary: route.websocket ? 'WebSocket connection' : `${route.method.toUpperCase()} ${route.path}`,
    tags: [tagForPath(route.path)],
    responses: route.websocket
      ? { 101: { description: 'Switching protocols' } }
      : route.method === 'delete'
        ? { 204: { description: 'Deleted' } }
        : { 200: { description: 'OK' } },
  }

  if (route.websocket) {
    operation.description = 'Upgrade to WebSocket for real-time streaming.'
  }

  if (parameters.length > 0) {
    operation.parameters = parameters
  }

  return operation
}

export function mergeOpenApiOverrides(
  ...overrides: Array<OpenApiOverrideMap | undefined>
): OpenApiOverrideMap {
  const merged: OpenApiOverrideMap = { ...OPERATION_OVERRIDES }

  for (const overrideMap of overrides) {
    if (!overrideMap) continue
    for (const [path, operations] of Object.entries(overrideMap)) {
      merged[path] = {
        ...(merged[path] ?? {}),
        ...operations,
      }
    }
  }

  return merged
}

export function mergeOpenApiComponents(
  ...components: Array<OpenApiComponentOverrides | undefined>
): OpenApiComponentOverrides {
  const merged: OpenApiComponentOverrides = {
    schemas: { ...(COMPONENT_OVERRIDES.schemas ?? {}) },
    parameters: { ...(COMPONENT_OVERRIDES.parameters ?? {}) },
  }

  for (const componentOverride of components) {
    if (!componentOverride) continue
    if (componentOverride.schemas) {
      merged.schemas = {
        ...(merged.schemas ?? {}),
        ...componentOverride.schemas,
      }
    }
    if (componentOverride.parameters) {
      merged.parameters = {
        ...(merged.parameters ?? {}),
        ...componentOverride.parameters,
      }
    }
  }

  return merged
}

export function openApiSchemaRef(name: string): OpenApiSchema {
  return { $ref: `#/components/schemas/${name}` }
}

export function openApiJsonResponse(
  schema: OpenApiSchema,
  description = 'OK',
): Record<string, unknown> {
  return {
    description,
    content: {
      'application/json': {
        schema,
      },
    },
  }
}

export function openApiJsonResponseRef(
  name: string,
  description = 'OK',
): Record<string, unknown> {
  return openApiJsonResponse(openApiSchemaRef(name), description)
}

export function openApiParameterRef(name: string): OpenApiParameterReference {
  return { $ref: `#/components/parameters/${name}` }
}

export function recordDiscoveredOpenApiRoute(
  routes: DiscoveredOpenApiRoute[],
  route: RouteDiscoveryInput,
): void {
  const methods = Array.isArray(route.method) ? route.method : [route.method]

  for (const rawMethod of methods) {
    const method = normalizeMethod(rawMethod)
    if (!method) continue

    const path = normalizePath(route.url)
    const exists = routes.some(
      (candidate) => candidate.method === method && candidate.path === path,
    )
    if (exists) continue

    routes.push({
      method,
      path,
      websocket: route.websocket === true,
    })
  }
}

export interface GenerateOpenApiSpecOptions {
  /**
   * Server URL to advertise in the `servers` field. Defaults to the
   * loopback bind so the spec renders correctly for local installs.
   * Pass the request's host/protocol when serving remote clients so
   * generated SDKs / swagger-ui pick up the reachable origin.
   */
  serverUrl?: string
  serverDescription?: string
}

export function generateOpenApiSpec(
  discoveredRoutes: DiscoveredOpenApiRoute[],
  operationOverrides: OpenApiOverrideMap = OPERATION_OVERRIDES,
  componentOverrides: OpenApiComponentOverrides = COMPONENT_OVERRIDES,
  specOptions: GenerateOpenApiSpecOptions = {},
): Record<string, unknown> {
  const sortedRoutes = [...discoveredRoutes].sort((left, right) => {
    const pathComparison = left.path.localeCompare(right.path)
    if (pathComparison !== 0) return pathComparison
    return METHOD_ORDER.indexOf(left.method) - METHOD_ORDER.indexOf(right.method)
  })

  const paths: Record<string, Record<string, unknown>> = {}

  for (const route of sortedRoutes) {
    const pathItem = paths[route.path] ?? {}
    const base = buildDefaultOperation(route)
    const override = operationOverrides[route.path]?.[route.method]
    pathItem[route.method] = override
      ? {
        ...base,
        ...override,
        parameters: mergeParameters(
          inferPathParameters(route.path),
          override.parameters,
        ),
        responses: override.responses ?? base.responses,
        security: override.security ?? base.security,
      }
      : base
    paths[route.path] = pathItem
  }

  return {
    openapi: '3.1.0',
    info: {
      title: 'sepilotd Daemon API',
      version: '0.2.10',
      description: 'AI agent daemon with GitHub-native control plane',
      license: { name: 'MIT' },
    },
    servers: [{
      url: specOptions.serverUrl ?? 'http://127.0.0.1:17600',
      description: specOptions.serverDescription ?? 'Local daemon',
    }],
    paths,
    components: {
      schemas: componentOverrides.schemas ?? {},
      parameters: componentOverrides.parameters ?? {},
      securitySchemes: {
        bearerAuth: { type: 'http', scheme: 'bearer' },
      },
    },
    security: [{ bearerAuth: [] }],
  }
}
