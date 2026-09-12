import type { ZodTypeAny } from 'zod'
import { zodToJsonSchema } from 'zod-to-json-schema'
import type {
  OpenApiComponentOverrides,
  OpenApiParameter,
  OpenApiSchema,
} from './openapi.js'

interface OpenApiZodParameterInput {
  name: string
  in: OpenApiParameter['in']
  required?: boolean
  schema: ZodTypeAny
}

interface OpenApiZodComponentInput {
  schemas?: Record<string, ZodTypeAny>
  parameters?: Record<string, OpenApiZodParameterInput>
}

function stripJsonSchemaMetadata(schema: OpenApiSchema): OpenApiSchema {
  const { $schema: _schema, definitions: _definitions, ...rest } = schema
  return rest
}

export function openApiSchemaFromZod(schema: ZodTypeAny): OpenApiSchema {
  return stripJsonSchemaMetadata(
    zodToJsonSchema(schema, {
      target: 'openApi3',
      $refStrategy: 'none',
    }) as OpenApiSchema,
  )
}

export function openApiParameterFromZod(input: OpenApiZodParameterInput): OpenApiParameter {
  return {
    name: input.name,
    in: input.in,
    required: input.required,
    schema: openApiSchemaFromZod(input.schema),
  }
}

export function openApiComponentsFromZod(
  input: OpenApiZodComponentInput,
): OpenApiComponentOverrides {
  return {
    schemas: input.schemas
      ? Object.fromEntries(
        Object.entries(input.schemas).map(([name, schema]) => [
          name,
          openApiSchemaFromZod(schema),
        ]),
      )
      : undefined,
    parameters: input.parameters
      ? Object.fromEntries(
        Object.entries(input.parameters).map(([name, parameter]) => [
          name,
          openApiParameterFromZod(parameter),
        ]),
      )
      : undefined,
  }
}
