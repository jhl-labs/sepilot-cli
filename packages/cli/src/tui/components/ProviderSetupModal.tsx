import { useEffect, useMemo, useState } from 'react'
import { Box, Text, useInput } from 'ink'
import { calculateVisibleWindow } from '../utils/layout.js'
import { isReturnKey } from '../utils/key.js'
import { colors, symbols } from '../theme.js'
import type { ProviderWizardPreset } from '../../utils/provider-presets.js'
import {
  PROVIDER_SETUP_STEP_LABELS,
  PROVIDER_SETUP_STEP_ORDER,
  connectionFieldsForPreset,
  describeApiKeySource,
  describeProviderPreset,
  filterModelSuggestions,
  filterProviderPresets,
  initialConnectionField,
  isFirstProviderSetupStep,
  presetUsesBaseUrl,
  type ProviderConnectionFieldKey,
  type ProviderSetupStep,
} from '../utils/provider-wizard.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'

export type { ProviderConnectionFieldKey, ProviderSetupStep } from '../utils/provider-wizard.js'

interface ProviderSetupModalProps {
  mode: 'new' | 'edit'
  step: ProviderSetupStep
  allowPresetSelection?: boolean
  applyToCurrentSession: boolean
  preset: ProviderWizardPreset | null
  presets: ProviderWizardPreset[]
  selectedPresetIndex: number
  providerId: string
  baseUrl: string
  apiKeyEnvVar: string
  apiKeyValue: string
  headersText?: string
  model: string
  modelSuggestions: string[]
  selectedModelSuggestionIndex: number
  loading: boolean
  validating: boolean
  saving: boolean
  error: string | null
  errorField?: ProviderConnectionFieldKey | null
  maxVisibleItems?: number
  onSelectPresetIndex: (index: number) => void
  onSelectPreset: (index: number) => void
  onProviderIdChange: (value: string) => void
  onBaseUrlChange: (value: string) => void
  onApiKeyEnvVarChange: (value: string) => void
  onApiKeyValueChange: (value: string) => void
  onHeadersTextChange?: (value: string) => void
  onModelChange: (value: string) => void
  onSelectModelSuggestionIndex: (index: number) => void
  onApplyModelSuggestion: (index: number) => void
  onToggleApplyToCurrentSession: () => void
  onNext: () => void
  onBack: () => void
  onClose: () => void
}

const LABEL_WIDTH = 13
const PRESET_LABEL_WIDTH = 18

function maskSecret(value: string): string {
  return '*'.repeat(Math.min(Math.max(value.length, 4), 24))
}

function StepIndicator({
  step,
  allowPresetSelection,
}: {
  step: ProviderSetupStep
  allowPresetSelection: boolean
}) {
  const steps = PROVIDER_SETUP_STEP_ORDER.filter(
    (entry) => entry !== 'preset' || allowPresetSelection,
  )
  const currentIndex = steps.indexOf(step)
  return (
    <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
      {steps.map((entry, index) => {
        const done = index < currentIndex
        const active = index === currentIndex
        return (
          <Text
            key={entry}
            color={active ? colors.text : done ? colors.success : colors.dimText}
            backgroundColor={active ? colors.primary : undefined}
            bold={active}
          >
            {' '}{done ? symbols.success : index + 1} {PROVIDER_SETUP_STEP_LABELS[entry]}{' '}
          </Text>
        )
      })}
    </Box>
  )
}

function SummaryRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
      <Box width={LABEL_WIDTH} flexShrink={0}>
        <Text color={colors.dimText}>{label}</Text>
      </Box>
      <Box flexGrow={1} flexShrink={1} minWidth={0}>
        <Text color={color ?? colors.text} wrap="truncate-end">{value}</Text>
      </Box>
    </Box>
  )
}

export function ProviderSetupModal({
  mode,
  step,
  allowPresetSelection = mode === 'new',
  applyToCurrentSession,
  preset,
  presets,
  selectedPresetIndex,
  providerId,
  baseUrl,
  apiKeyEnvVar,
  apiKeyValue,
  headersText = '',
  model,
  modelSuggestions,
  selectedModelSuggestionIndex,
  loading,
  validating,
  saving,
  error,
  errorField = null,
  maxVisibleItems = 8,
  onSelectPresetIndex,
  onSelectPreset,
  onProviderIdChange,
  onBaseUrlChange,
  onApiKeyEnvVarChange,
  onApiKeyValueChange,
  onHeadersTextChange = () => {},
  onModelChange,
  onSelectModelSuggestionIndex,
  onApplyModelSuggestion,
  onToggleApplyToCurrentSession,
  onNext,
  onBack,
  onClose,
}: ProviderSetupModalProps) {
  const statusBusy = loading || validating || saving
  const isFirstStep = isFirstProviderSetupStep(step, allowPresetSelection)

  const [presetQuery, setPresetQuery] = useState('')
  const [presetHighlight, setPresetHighlight] = useState(0)
  const filteredPresets = useMemo(
    () => filterProviderPresets(presets, presetQuery),
    [presetQuery, presets],
  )
  const presetHighlightIndex = filteredPresets.length === 0
    ? 0
    : Math.max(0, Math.min(presetHighlight, filteredPresets.length - 1))
  const presetWindow = useMemo(
    () => calculateVisibleWindow(filteredPresets.length, presetHighlightIndex, maxVisibleItems),
    [filteredPresets.length, maxVisibleItems, presetHighlightIndex],
  )

  useEffect(() => {
    if (step !== 'preset') return
    setPresetQuery('')
    const initial = presets[selectedPresetIndex]
    setPresetHighlight(Math.max(0, initial ? presets.indexOf(initial) : 0))
  }, [step])

  const movePresetHighlight = (next: number) => {
    const clamped = Math.max(0, Math.min(next, filteredPresets.length - 1))
    setPresetHighlight(clamped)
    const target = filteredPresets[clamped]
    if (target) onSelectPresetIndex(presets.indexOf(target))
  }

  const fields = useMemo(() => connectionFieldsForPreset(preset), [preset])
  const [focusedField, setFocusedField] = useState<ProviderConnectionFieldKey>(
    () => initialConnectionField(preset),
  )
  const presetType = preset?.type ?? null
  useEffect(() => {
    if (step === 'connection') setFocusedField(initialConnectionField(preset))
  }, [presetType, step])
  useEffect(() => {
    if (errorField && fields.some((field) => field.key === errorField)) {
      setFocusedField(errorField)
    }
  }, [errorField, fields])
  const focusedFieldIndex = Math.max(0, fields.findIndex((field) => field.key === focusedField))
  const moveFocus = (delta: number) => {
    if (fields.length === 0) return
    const next = (focusedFieldIndex + delta + fields.length) % fields.length
    setFocusedField(fields[next]!.key)
  }
  const fieldValues: Record<ProviderConnectionFieldKey, string> = {
    providerId,
    baseUrl,
    apiKeyValue,
    apiKeyEnvVar,
    headers: headersText,
  }
  const fieldSetters: Record<ProviderConnectionFieldKey, (value: string) => void> = {
    providerId: onProviderIdChange,
    baseUrl: onBaseUrlChange,
    apiKeyValue: onApiKeyValueChange,
    apiKeyEnvVar: onApiKeyEnvVarChange,
    headers: onHeadersTextChange,
  }

  const trimmedModel = model.trim()
  const filteredModels = useMemo(
    () => filterModelSuggestions(modelSuggestions, model),
    [model, modelSuggestions],
  )
  const exactModelMatch = trimmedModel.length > 0 && filteredModels.includes(trimmedModel)
  const showTypedRow = trimmedModel.length > 0 && !exactModelMatch
  const modelRowCount = filteredModels.length + (showTypedRow ? 1 : 0)
  const [modelHighlight, setModelHighlight] = useState(0)
  const modelHighlightIndex = modelRowCount === 0
    ? 0
    : Math.max(0, Math.min(modelHighlight, modelRowCount - 1))
  const modelWindow = useMemo(
    () => calculateVisibleWindow(modelRowCount, modelHighlightIndex, maxVisibleItems),
    [maxVisibleItems, modelHighlightIndex, modelRowCount],
  )
  const suggestionAtRow = (row: number): string | null => {
    const index = showTypedRow ? row - 1 : row
    return index >= 0 ? (filteredModels[index] ?? null) : null
  }
  useEffect(() => {
    if (step !== 'model') return
    const filtered = filterModelSuggestions(modelSuggestions, model)
    const typed = model.trim()
    const exactIndex = typed ? filtered.indexOf(typed) : -1
    const preferred = modelSuggestions[selectedModelSuggestionIndex]
    const preferredIndex = preferred ? filtered.indexOf(preferred) : -1
    const typedRow = typed.length > 0 && exactIndex < 0 ? 1 : 0
    const seed = exactIndex >= 0 ? exactIndex : preferredIndex >= 0 ? preferredIndex : 0
    setModelHighlight(seed + typedRow)
  }, [step, modelSuggestions])
  const moveModelHighlight = (next: number) => {
    const clamped = Math.max(0, Math.min(next, modelRowCount - 1))
    setModelHighlight(clamped)
    const suggestion = suggestionAtRow(clamped)
    if (suggestion) onSelectModelSuggestionIndex(modelSuggestions.indexOf(suggestion))
  }
  const handleModelChange = (value: string) => {
    onModelChange(value)
    const filtered = filterModelSuggestions(modelSuggestions, value)
    const typedRow = value.trim().length > 0 && !filtered.includes(value.trim()) ? 1 : 0
    setModelHighlight(filtered.length > 0 ? typedRow : 0)
  }

  useInput((input, key) => {
    if (statusBusy) return

    if (key.escape) {
      if (isFirstStep) onClose()
      else onBack()
      return
    }

    if (step === 'preset') {
      if (key.upArrow) { movePresetHighlight(presetHighlightIndex - 1); return }
      if (key.downArrow) { movePresetHighlight(presetHighlightIndex + 1); return }
      if (isReturnKey(input, key) || key.tab) {
        const target = filteredPresets[presetHighlightIndex]
        if (target) onSelectPreset(presets.indexOf(target))
      }
      return
    }

    if (step === 'connection') {
      if (key.tab && key.shift) { moveFocus(-1); return }
      if (key.tab || input === '\t') { moveFocus(1); return }
      if (key.upArrow) { moveFocus(-1); return }
      if (key.downArrow) { moveFocus(1); return }
      if (isReturnKey(input, key)) onNext()
      return
    }

    if (step === 'model') {
      if (key.upArrow) { moveModelHighlight(modelHighlightIndex - 1); return }
      if (key.downArrow) { moveModelHighlight(modelHighlightIndex + 1); return }
      if (key.tab || input === '\t') {
        const suggestion = suggestionAtRow(modelHighlightIndex)
        if (suggestion) onApplyModelSuggestion(modelSuggestions.indexOf(suggestion))
        return
      }
      if (isReturnKey(input, key)) {
        const suggestion = suggestionAtRow(modelHighlightIndex)
        if (suggestion && suggestion !== trimmedModel) {
          onApplyModelSuggestion(modelSuggestions.indexOf(suggestion))
          return
        }
        onNext()
      }
      return
    }

    if (step === 'confirm') {
      if (input === ' ' || (input === 's' && key.ctrl)) {
        onToggleApplyToCurrentSession()
        return
      }
      if (isReturnKey(input, key)) onNext()
    }
  })

  const statusLine = error
    ? { text: error, color: colors.error }
    : loading
      ? { text: 'Discovering models on the endpoint...', color: colors.info }
      : validating
        ? { text: 'Testing the connection with the selected model...', color: colors.info }
        : saving
          ? { text: 'Saving provider configuration...', color: colors.info }
          : step === 'connection'
            ? { text: fields.find((field) => field.key === focusedField)?.hint ?? '', color: colors.dimText }
            : step === 'model'
              ? {
                  text: modelSuggestions.length > 0
                    ? `${modelSuggestions.length} model${modelSuggestions.length === 1 ? '' : 's'} available. Type to filter, or enter any model id.`
                    : 'No models were discovered. Enter the model id manually.',
                  color: colors.dimText,
                }
              : step === 'confirm'
                ? { text: 'Saving runs a live connection test first; nothing is written if it fails.', color: colors.dimText }
                : { text: 'Type to filter providers.', color: colors.dimText }

  const backHint = isFirstStep ? 'Esc close' : 'Esc back'
  const footer = statusBusy
    ? 'Please wait...'
    : step === 'preset'
      ? `↑↓ move  Enter select  ${backHint}`
      : step === 'connection'
        ? `Tab/↑↓ fields  Enter continue  ${backHint}`
        : step === 'model'
          ? `↑↓ move  Enter pick/continue  Tab fill  ${backHint}`
          : `Enter test & save  Space toggle session  ${backHint}`

  const contextLine = mode === 'edit'
    ? `Editing ${providerId}${preset ? ` (${preset.label})` : ''}`
    : preset && step !== 'preset'
      ? `Adding ${preset.label} provider as "${providerId || preset.defaultProviderId || preset.type}"`
      : 'Add a provider to the daemon.'

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
      marginY={1}
      width="100%"
      minWidth={0}
    >
      <Text color={colors.primary} bold>Provider Setup</Text>
      <StepIndicator step={step} allowPresetSelection={allowPresetSelection} />
      <Text color={colors.dimText} wrap="truncate-end">{contextLine}</Text>

      {step === 'preset' && (
        <>
          <Box gap={1}>
            <Text color={colors.dimText}>search</Text>
            <ControlSafeTextInput
              value={presetQuery}
              onChange={(value) => { setPresetQuery(value); setPresetHighlight(0) }}
              placeholder="openai, ollama, compatible..."
              focus={!statusBusy}
            />
          </Box>
          {filteredPresets.length === 0 ? (
            <Text color={colors.dimText}>No provider matches "{presetQuery}".</Text>
          ) : (
            filteredPresets.slice(presetWindow.start, presetWindow.end).map((item, index) => {
              const selected = presetWindow.start + index === presetHighlightIndex
              return (
                <Box key={item.type} gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
                  <Text color={selected ? colors.primary : colors.dimText}>{selected ? '›' : ' '}</Text>
                  <Box width={PRESET_LABEL_WIDTH} flexShrink={0}>
                    <Text color={selected ? colors.primary : colors.text} bold={selected} wrap="truncate-end">
                      {item.label}
                    </Text>
                  </Box>
                  <Box flexGrow={1} flexShrink={1} minWidth={0}>
                    <Text color={colors.dimText} wrap="truncate-end">{describeProviderPreset(item)}</Text>
                  </Box>
                </Box>
              )
            })
          )}
        </>
      )}

      {step === 'connection' && fields.map((field) => {
        const focused = field.key === focusedField
        const value = fieldValues[field.key]
        const hasError = errorField === field.key && Boolean(error)
        const labelColor = hasError ? colors.error : focused ? colors.primary : colors.text
        const display = value
          ? field.secret ? maskSecret(value) : value
          : field.optional ? '(optional)' : field.placeholder
        return (
          <Box key={field.key} gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
            <Text color={labelColor}>{focused ? '›' : ' '}</Text>
            <Box width={LABEL_WIDTH} flexShrink={0}>
              <Text color={labelColor} bold={focused} wrap="truncate-end">{field.label}</Text>
            </Box>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              {focused ? (
                <ControlSafeTextInput
                  key={field.key}
                  value={value}
                  onChange={fieldSetters[field.key]}
                  placeholder={field.placeholder}
                  mask={field.secret ? '*' : undefined}
                  focus={!statusBusy}
                />
              ) : (
                <Text color={value ? colors.text : colors.dimText} wrap="truncate-end">{display}</Text>
              )}
            </Box>
          </Box>
        )
      })}

      {step === 'model' && (
        <>
          <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
            <Text color={colors.primary}>›</Text>
            <Box width={LABEL_WIDTH} flexShrink={0}>
              <Text color={colors.primary} bold>Model</Text>
            </Box>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <ControlSafeTextInput
                value={model}
                onChange={handleModelChange}
                placeholder={preset?.suggestedModels?.[0] ?? 'model-id'}
                focus={!statusBusy}
              />
            </Box>
          </Box>
          {Array.from({ length: modelWindow.end - modelWindow.start }, (_, offset) => {
            const row = modelWindow.start + offset
            const selected = row === modelHighlightIndex
            const suggestion = suggestionAtRow(row)
            const label = suggestion ?? `Use "${trimmedModel}" as typed`
            return (
              <Box key={suggestion ?? '__typed__'} gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
                <Text color={selected ? colors.primary : colors.dimText}>{selected ? '›' : ' '}</Text>
                <Box flexGrow={1} flexShrink={1} minWidth={0}>
                  <Text
                    color={selected ? colors.primary : suggestion ? colors.text : colors.dimText}
                    bold={selected}
                    wrap="truncate-end"
                  >
                    {label}
                  </Text>
                </Box>
                {suggestion && suggestion === trimmedModel ? (
                  <Text color={colors.success}>[current]</Text>
                ) : null}
              </Box>
            )
          })}
        </>
      )}

      {step === 'confirm' && (
        <>
          <SummaryRow label="Provider" value={`${providerId}${preset ? ` (${preset.label})` : ''}`} />
          {presetUsesBaseUrl(preset) ? <SummaryRow label="Base URL" value={baseUrl} /> : null}
          {(() => {
            const source = describeApiKeySource({ preset, apiKeyEnvVar, apiKeyValue })
            return source ? <SummaryRow label="API key" value={source} /> : null
          })()}
          {preset?.type === 'custom' ? (
            <SummaryRow label="Headers" value={headersText.trim() ? headersText.trim() : 'none'} />
          ) : null}
          <SummaryRow label="Model" value={model} />
          <SummaryRow label="Default" value={`${providerId}/${model} becomes the daemon default`} color={colors.muted} />
          <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
            <Text color={applyToCurrentSession ? colors.success : colors.dimText}>
              [{applyToCurrentSession ? 'x' : ' '}]
            </Text>
            <Text color={colors.text} wrap="truncate-end">
              Switch this session to {providerId}/{model} after saving
            </Text>
          </Box>
        </>
      )}

      <Text color={statusLine.color} wrap="truncate-end">{statusLine.text || ' '}</Text>
      <Text color={colors.dimText}>{footer}</Text>
    </Box>
  )
}
