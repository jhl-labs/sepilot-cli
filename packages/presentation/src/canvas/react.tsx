import { useState } from 'react'
import { isSafeA2UIImageSrc, type A2UIComponent, type A2UIForm } from '@sepilotd/core'

export interface A2UICanvasProps {
  components: A2UIComponent[]
  className?: string
  onFormSubmit?: (data: Record<string, string>) => void
  emptyState?: string | null
}

const STACK_STYLE = {
  display: 'grid',
  gap: '12px',
} as const

const SECTION_TITLE_STYLE = {
  fontSize: '0.95rem',
  fontWeight: 600,
  marginBottom: '8px',
} as const

const CARD_STYLE = {
  border: '1px solid color-mix(in srgb, currentColor 14%, transparent)',
  borderRadius: '12px',
  padding: '12px',
  background: 'color-mix(in srgb, Canvas 94%, currentColor 6%)',
} as const

const MUTED_TEXT_STYLE = {
  color: 'color-mix(in srgb, currentColor 62%, transparent)',
  fontSize: '0.9rem',
} as const

const INPUT_STYLE = {
  width: '100%',
  borderRadius: '10px',
  border: '1px solid color-mix(in srgb, currentColor 18%, transparent)',
  padding: '8px 10px',
  background: 'Canvas',
  color: 'inherit',
  font: 'inherit',
} as const

const BUTTON_STYLE = {
  border: 'none',
  borderRadius: '10px',
  padding: '8px 12px',
  background: 'color-mix(in srgb, #2563eb 88%, white 12%)',
  color: 'white',
  cursor: 'pointer',
  font: 'inherit',
  fontWeight: 600,
} as const

export function A2UICanvas({
  components,
  className,
  onFormSubmit,
  emptyState = null,
}: A2UICanvasProps) {
  if (components.length === 0) {
    return emptyState ? <div className={className}>{emptyState}</div> : null
  }

  return (
    <div className={className} style={STACK_STYLE}>
      {components.map((component, index) => (
        <A2UICanvasComponent
          key={`${component.type}-${index}`}
          component={component}
          onFormSubmit={onFormSubmit}
        />
      ))}
    </div>
  )
}

function A2UICanvasComponent({
  component,
  onFormSubmit,
}: {
  component: A2UIComponent
  onFormSubmit?: (data: Record<string, string>) => void
}) {
  switch (component.type) {
    case 'text':
      return (
        <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
          {component.content}
        </div>
      )

    case 'table':
      return (
        <section>
          {component.title && (
            <div style={SECTION_TITLE_STYLE}>{component.title}</div>
          )}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {component.headers.map((header, index) => (
                    <th
                      key={`${header}-${index}`}
                      style={{
                        textAlign: 'left',
                        padding: '8px 10px',
                        borderBottom: '1px solid color-mix(in srgb, currentColor 18%, transparent)',
                      }}
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {component.rows.map((row, rowIndex) => (
                  <tr key={`row-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td
                        key={`cell-${rowIndex}-${cellIndex}`}
                        style={{
                          padding: '8px 10px',
                          borderBottom: '1px solid color-mix(in srgb, currentColor 10%, transparent)',
                          verticalAlign: 'top',
                        }}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )

    case 'code':
      return (
        <section>
          {component.title && (
            <div style={{ ...SECTION_TITLE_STYLE, ...MUTED_TEXT_STYLE }}>
              {component.title}
            </div>
          )}
          <pre
            style={{
              ...CARD_STYLE,
              margin: 0,
              overflowX: 'auto',
              fontSize: '0.9rem',
            }}
          >
            <code>{component.code}</code>
          </pre>
        </section>
      )

    case 'list': {
      const Tag = component.ordered ? 'ol' : 'ul'
      return (
        <section>
          {component.title && (
            <div style={SECTION_TITLE_STYLE}>{component.title}</div>
          )}
          <Tag style={{ margin: 0, paddingLeft: '20px' }}>
            {component.items.map((item, index) => (
              <li key={`${item.text}-${index}`} style={{ marginBottom: '8px' }}>
                <div>
                  {item.icon ? `${item.icon} ` : null}
                  {item.text}
                </div>
                {item.description && (
                  <div style={MUTED_TEXT_STYLE}>{item.description}</div>
                )}
              </li>
            ))}
          </Tag>
        </section>
      )
    }

    case 'progress':
      return (
        <section>
          {component.title && (
            <div style={SECTION_TITLE_STYLE}>{component.title}</div>
          )}
          <div style={{ display: 'grid', gap: '10px' }}>
            {component.steps.map((step, index) => (
              <div
                key={`${step.label}-${index}`}
                style={{ display: 'grid', gridTemplateColumns: '28px 1fr', gap: '10px', alignItems: 'center' }}
              >
                <div
                  aria-hidden="true"
                  style={{
                    width: '28px',
                    height: '28px',
                    borderRadius: '999px',
                    display: 'grid',
                    placeItems: 'center',
                    background:
                      step.status === 'done'
                        ? '#16a34a'
                        : step.status === 'active'
                          ? '#2563eb'
                          : 'color-mix(in srgb, currentColor 14%, transparent)',
                    color: step.status === 'pending' ? 'inherit' : 'white',
                    fontWeight: 700,
                    fontSize: '0.85rem',
                  }}
                >
                  {step.status === 'done' ? '✓' : index + 1}
                </div>
                <div>
                  <div>{step.label}</div>
                  <div style={MUTED_TEXT_STYLE}>{labelForProgressStatus(step.status)}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )

    case 'image':
      if (!isSafeA2UIImageSrc(component.src)) {
        return (
          <div role="note" style={{ ...CARD_STYLE, ...MUTED_TEXT_STYLE }}>
            Image source blocked
          </div>
        )
      }
      return (
        <img
          src={component.src}
          alt={component.alt ?? ''}
          width={component.width}
          height={component.height}
          style={{
            maxWidth: '100%',
            height: 'auto',
            borderRadius: '12px',
            border: '1px solid color-mix(in srgb, currentColor 12%, transparent)',
          }}
        />
      )

    case 'chart':
      return (
        <section style={CARD_STYLE}>
          {component.title && (
            <div style={SECTION_TITLE_STYLE}>{component.title}</div>
          )}
          <div>{labelForChartType(component.chartType)} chart</div>
          <div style={MUTED_TEXT_STYLE}>
            {component.data.labels.length} labels · {component.data.datasets.length} datasets
          </div>
        </section>
      )

    case 'form':
      return (
        <A2UICanvasForm
          form={component}
          onSubmit={onFormSubmit}
        />
      )
  }
}

function A2UICanvasForm({
  form,
  onSubmit,
}: {
  form: A2UIForm
  onSubmit?: (data: Record<string, string>) => void
}) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      form.fields.map((field) => [field.name, field.defaultValue ?? '']),
    ),
  )

  return (
    <form
      style={CARD_STYLE}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit?.(values)
      }}
    >
      {form.title && (
        <div style={SECTION_TITLE_STYLE}>{form.title}</div>
      )}

      <div style={STACK_STYLE}>
        {form.fields.map((field) => (
          <label key={field.name} style={{ display: 'grid', gap: '6px' }}>
            <span style={MUTED_TEXT_STYLE}>{field.label}</span>
            <A2UICanvasField
              field={field}
              value={values[field.name] ?? ''}
              onChange={(nextValue) => {
                setValues((prev) => ({ ...prev, [field.name]: nextValue }))
              }}
            />
          </label>
        ))}
      </div>

      <div style={{ marginTop: '12px' }}>
        <button type="submit" style={BUTTON_STYLE}>
          {form.submitLabel ?? 'Submit'}
        </button>
      </div>
    </form>
  )
}

function A2UICanvasField({
  field,
  value,
  onChange,
}: {
  field: A2UIForm['fields'][number]
  value: string
  onChange: (nextValue: string) => void
}) {
  switch (field.type) {
    case 'textarea':
      return (
        <textarea
          value={value}
          required={field.required}
          onChange={(event) => onChange(event.target.value)}
          style={{ ...INPUT_STYLE, minHeight: '88px', resize: 'vertical' }}
        />
      )
    case 'select':
      return (
        <select
          value={value}
          required={field.required}
          onChange={(event) => onChange(event.target.value)}
          style={INPUT_STYLE}
        >
          {!field.required && <option value="">Select…</option>}
          {(field.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      )
    case 'checkbox':
      return (
        <label style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={value === 'true'}
            onChange={(event) => onChange(event.target.checked ? 'true' : 'false')}
          />
          <span style={MUTED_TEXT_STYLE}>Enabled</span>
        </label>
      )
    default:
      return (
        <input
          type={field.type === 'number' ? 'number' : 'text'}
          value={value}
          required={field.required}
          onChange={(event) => onChange(event.target.value)}
          style={INPUT_STYLE}
        />
      )
  }
}

function labelForChartType(chartType: 'bar' | 'line' | 'pie' | 'scatter'): string {
  switch (chartType) {
    case 'bar':
      return 'Bar'
    case 'line':
      return 'Line'
    case 'pie':
      return 'Pie'
    case 'scatter':
      return 'Scatter'
  }
}

function labelForProgressStatus(status: 'done' | 'active' | 'pending'): string {
  switch (status) {
    case 'done':
      return 'Completed'
    case 'active':
      return 'In progress'
    case 'pending':
      return 'Pending'
  }
}
