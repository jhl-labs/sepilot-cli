import { useEffect, useMemo, useState } from 'react'
import { DEFAULT_OPENAI_COMPATIBLE_BASE_URL } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { calculateVisibleWindow } from '../utils/layout.js'
import { isReturnKey } from '../utils/key.js'
import { colors, symbols } from '../theme.js'
import type { ProviderWizardPreset } from '../../utils/provider-presets.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'

export type ProviderSetupStep =
  | 'preset'
  | 'providerId'
  | 'baseUrl'
  | 'apiKeyEnv'
  | 'apiKeyValue'
  | 'headers'
  | 'model'
  | 'confirm'

type ProviderSetupConfirmAction = 'back' | 'save'

interface ProviderSetupModalProps {
  mode: 'new' | 'edit'
  step: ProviderSetupStep
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

function fieldValueForStep(
  step: ProviderSetupStep,
  values: {
    providerId: string
    baseUrl: string
    apiKeyEnvVar: string
    apiKeyValue: string
    headersText: string
    model: string
  },
): string {
  switch (step) {
    case 'providerId':
      return values.providerId
    case 'baseUrl':
      return values.baseUrl
    case 'apiKeyEnv':
      return values.apiKeyEnvVar
    case 'apiKeyValue':
      return values.apiKeyValue
    case 'headers':
      return values.headersText
    case 'model':
      return values.model
    default:
      return ''
  }
}

function footerHintForStep(
  step: ProviderSetupStep,
  options: {
    error: string | null
    statusBusy: boolean
    hasModelSuggestions: boolean
  },
): string {
  if (options.statusBusy) {
    return 'Please wait'
  }

  if (options.error) {
    return step === 'confirm'
      ? 'Enter retry  Backspace edit  Esc close'
      : 'Enter retry  Backspace back  Esc close'
  }

  switch (step) {
    case 'preset':
      return '↑/↓ move  Enter select  1-9 quick pick  Esc close'
    case 'confirm':
      return '←/→ move  Enter select  Tab toggle  Ctrl+S session sync  Backspace edit  Esc close'
    case 'model':
      return options.hasModelSuggestions
        ? '↑/↓ suggestions  Tab fill  Ctrl+U clear  Enter next  Backspace back when empty  Esc close'
        : 'Ctrl+U clear  Enter next  Backspace back when empty  Esc close'
    default:
      return 'Enter next  Backspace back when empty  Esc close'
  }
}

export function ProviderSetupModal({
  mode,
  step,
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
  const [selectedConfirmActionIndex, setSelectedConfirmActionIndex] = useState(1)
  const statusBusy = loading || validating || saving
  const selectedConfirmAction: ProviderSetupConfirmAction =
    selectedConfirmActionIndex === 0 ? 'back' : 'save'
  const fieldValue = fieldValueForStep(step, {
    providerId,
    baseUrl,
    apiKeyEnvVar,
    apiKeyValue,
    headersText,
    model,
  })
  const clampedPresetIndex = presets.length === 0
    ? 0
    : Math.max(0, Math.min(selectedPresetIndex, presets.length - 1))
  const clampedSuggestionIndex = modelSuggestions.length === 0
    ? 0
    : Math.max(0, Math.min(selectedModelSuggestionIndex, modelSuggestions.length - 1))

  const presetWindow = useMemo(
    () => calculateVisibleWindow(presets.length, clampedPresetIndex, maxVisibleItems),
    [clampedPresetIndex, maxVisibleItems, presets.length],
  )
  const visiblePresets = presets.slice(presetWindow.start, presetWindow.end)

  const suggestionWindow = useMemo(
    () => calculateVisibleWindow(modelSuggestions.length, clampedSuggestionIndex, maxVisibleItems),
    [clampedSuggestionIndex, maxVisibleItems, modelSuggestions.length],
  )
  const visibleSuggestions = modelSuggestions.slice(
    suggestionWindow.start,
    suggestionWindow.end,
  )

  useEffect(() => {
    if (step === 'confirm') {
      setSelectedConfirmActionIndex(1)
    }
  }, [step])

  const footerHint = footerHintForStep(step, {
    error,
    statusBusy,
    hasModelSuggestions: modelSuggestions.length > 0,
  })
  const isOpenAiCompatible = preset?.type === 'custom'

  useInput((input, key) => {
    if (statusBusy) {
      return
    }

    if (key.escape) {
      onClose()
      return
    }

    if (step === 'preset') {
      if (/^[1-9]$/.test(input)) {
        const quickIndex = Number(input) - 1
        if (quickIndex < presets.length) {
          onSelectPresetIndex(quickIndex)
          onSelectPreset(quickIndex)
        }
        return
      }
      if (key.upArrow) {
        onSelectPresetIndex(Math.max(0, clampedPresetIndex - 1))
        return
      }
      if (key.downArrow) {
        onSelectPresetIndex(Math.min(presets.length - 1, clampedPresetIndex + 1))
        return
      }
      if ((isReturnKey(input, key) || input === '\t') && presets[clampedPresetIndex]) {
        onSelectPreset(clampedPresetIndex)
      }
      return
    }

    if (step === 'confirm') {
      if (input === 's' && key.ctrl) {
        onToggleApplyToCurrentSession()
        return
      }
      if (key.backspace || key.delete) {
        onBack()
        return
      }
      if (key.leftArrow || key.upArrow) {
        setSelectedConfirmActionIndex(0)
        return
      }
      if (key.rightArrow || key.downArrow) {
        setSelectedConfirmActionIndex(1)
        return
      }
      if (key.tab) {
        setSelectedConfirmActionIndex((current) => current === 0 ? 1 : 0)
        return
      }
      if (isReturnKey(input, key)) {
        if (selectedConfirmAction === 'save') {
          onNext()
        } else {
          onBack()
        }
      }
      return
    }

    if (step === 'model') {
      if (key.upArrow) {
        onSelectModelSuggestionIndex(Math.max(0, clampedSuggestionIndex - 1))
        return
      }
      if (key.downArrow) {
        onSelectModelSuggestionIndex(
          Math.min(modelSuggestions.length - 1, clampedSuggestionIndex + 1),
        )
        return
      }
      if (key.tab) {
        if (modelSuggestions[clampedSuggestionIndex]) {
          onApplyModelSuggestion(clampedSuggestionIndex)
        }
        return
      }
      if (isReturnKey(input, key)) {
        onNext()
        return
      }
    }

    if ((key.backspace || key.delete) && fieldValue.length === 0) {
      onBack()
      return
    }

    if (isReturnKey(input, key)) {
      onNext()
    }
  })

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
      <Text color={colors.primary} bold>
        Provider Setup
      </Text>
      <Text color={colors.dimText}>
        {mode === 'edit' ? 'Edit the active provider configuration.' : 'Add or replace a provider configuration.'}
      </Text>
      {preset ? (
        <Text color={colors.dimText}>
          {preset.label} {symbols.separator} {providerId || preset.type}
        </Text>
      ) : null}
      {step === 'confirm' && (
        <Text color={colors.dimText}>
          save target {symbols.separator} {applyToCurrentSession ? 'daemon default + current session' : 'daemon default only'}
        </Text>
      )}
      {error ? (
        <>
          <Text color={colors.error}>{error}</Text>
          <Text color={colors.dimText}>
            Fix the provider values and retry. Validation happens before any config update is saved.
          </Text>
        </>
      ) : null}
      {loading ? (
        <Text color={colors.dimText}>Loading provider configuration...</Text>
      ) : null}
      {validating ? (
        <Text color={colors.dimText}>Testing provider connection and model...</Text>
      ) : null}
      {saving ? (
        <Text color={colors.dimText}>Saving provider configuration...</Text>
      ) : null}

      {!statusBusy && step === 'preset' && (
        <>
          <Text color={colors.text}>Select a provider type</Text>
          {visiblePresets.map((item, index) => {
            const absoluteIndex = presetWindow.start + index
            const selected = absoluteIndex === clampedPresetIndex
            return (
              <Box key={item.type} gap={1} width="100%" minWidth={0}>
                <Text
                  color={selected ? colors.text : colors.dimText}
                  backgroundColor={selected ? colors.primary : undefined}
                  bold={selected}
                >
                  {' '}{absoluteIndex + 1}{' '}
                </Text>
                <Box flexGrow={1} flexShrink={1} minWidth={0}>
                  <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
                    {item.label} {symbols.separator} {item.apiKeyEnvVar
                      ? `${item.apiKeyEnvVar} ${symbols.separator} ${item.suggestedModels?.slice(0, 2).join(', ') ?? 'manual model'}`
                      : item.type === 'custom'
                        ? [
                            item.defaultBaseUrl ?? 'OpenAI-compatible /v1 endpoint',
                            'API key optional',
                            'custom headers supported',
                          ].join(` ${symbols.separator} `)
                        : 'Local provider'}
                  </Text>
                </Box>
              </Box>
            )
          })}
        </>
      )}

      {!statusBusy && step === 'providerId' && (
        <>
          <Text color={colors.text}>Provider id</Text>
          <ControlSafeTextInput
            value={providerId}
            onChange={onProviderIdChange}
            placeholder={preset?.type ?? 'provider-id'}
          />
          <Text color={colors.dimText}>
            This id is used in `/provider`, `/model`, and session state.
          </Text>
        </>
      )}

      {!statusBusy && step === 'baseUrl' && (
        <>
          <Text color={colors.text}>
            {isOpenAiCompatible ? 'OpenAI-compatible base URL' : 'Ollama base URL'}
          </Text>
          <ControlSafeTextInput
            value={baseUrl}
            onChange={onBaseUrlChange}
            placeholder={
              baseUrl
              || (isOpenAiCompatible
                ? DEFAULT_OPENAI_COMPATIBLE_BASE_URL
                : 'Set OLLAMA_HOST or enter a URL')
            }
          />
          {isOpenAiCompatible ? (
            <>
              <Text color={colors.dimText}>
                Use any OpenAI-compatible /v1 endpoint, for example Ollama, LM Studio,
                vLLM, or a hosted gateway.
              </Text>
              <Text color={colors.dimText}>
                Next you can configure an API key and custom HTTP headers, or leave both blank for a local server.
              </Text>
            </>
          ) : (
            <Text color={colors.dimText}>
              The saved provider will use this daemon-side Ollama endpoint.
            </Text>
          )}
        </>
      )}

      {!statusBusy && step === 'apiKeyEnv' && (
        <>
          <Text color={colors.text}>
            {preset?.label ?? 'Provider'} API key environment variable{isOpenAiCompatible ? ' (optional)' : ''}
          </Text>
          <ControlSafeTextInput
            value={apiKeyEnvVar}
            onChange={onApiKeyEnvVarChange}
            placeholder={preset?.apiKeyEnvVar ?? 'API_KEY'}
          />
          <Text color={colors.dimText}>
            {isOpenAiCompatible && !apiKeyEnvVar ? (
              'Leave blank to continue without authentication.'
            ) : (
              <>
                Enter only the variable name. It is stored as {'${'}{apiKeyEnvVar || preset?.apiKeyEnvVar || 'API_KEY'}{'}'} in `config.yaml`.
              </>
            )}
          </Text>
          <Text color={colors.dimText}>
            This step is only for an existing shell or service environment variable, not the API key value.
          </Text>
        </>
      )}

      {!statusBusy && step === 'apiKeyValue' && (
        <>
          <Text color={colors.text}>{preset?.label ?? 'Provider'} API key (optional)</Text>
          <ControlSafeTextInput
            value={apiKeyValue}
            onChange={onApiKeyValueChange}
            placeholder="Paste an API key, or leave blank to use an environment variable"
            mask="*"
          />
          <Text color={colors.dimText}>
            The API key is treated as an opaque value; environment-variable naming rules do not apply to it.
          </Text>
          <Text color={colors.dimText}>
            A value is saved in `~/.sepilotd/.env` only after validation succeeds. Leave it blank to configure an environment variable next.
          </Text>
        </>
      )}

      {!statusBusy && step === 'headers' && (
        <>
          <Text color={colors.text}>Custom HTTP headers (optional JSON)</Text>
          <ControlSafeTextInput
            value={headersText}
            onChange={onHeadersTextChange}
            placeholder={'{"X-API-Key":"${MY_API_KEY}","X-Tenant":"team-a"}'}
          />
          <Text color={colors.dimText}>
            Header names and string values are saved on this provider and sent with model discovery and chat requests.
          </Text>
          <Text color={colors.dimText}>
            Prefer {'${ENV_VAR}'} references for secret header values; leave blank for no custom headers.
          </Text>
        </>
      )}

      {!statusBusy && step === 'model' && (
        <>
          <Text color={colors.text}>Default model</Text>
          <ControlSafeTextInput
            value={model}
            onChange={onModelChange}
            placeholder={preset?.suggestedModels?.[0] ?? 'model-name'}
          />
          {model.trim() && modelSuggestions.length > 0 && !modelSuggestions.includes(model.trim()) ? (
            <Text color={colors.warning}>
              This model is not advertised by the endpoint. Choose a suggestion with Tab or press Ctrl+U to replace it.
            </Text>
          ) : null}
          {visibleSuggestions.length > 0 ? (
            <>
              <Text color={colors.dimText}>Suggestions</Text>
              {visibleSuggestions.map((item, index) => {
                const absoluteIndex = suggestionWindow.start + index
                const selected = absoluteIndex === clampedSuggestionIndex
                return (
                  <Box key={item} gap={1} width="100%" minWidth={0}>
                    <Text
                      color={selected ? colors.text : colors.dimText}
                      backgroundColor={selected ? colors.primary : undefined}
                      bold={selected}
                    >
                      {' '}{absoluteIndex + 1}{' '}
                    </Text>
                    <Box flexGrow={1} flexShrink={1} minWidth={0}>
                      <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
                        {item}
                      </Text>
                    </Box>
                  </Box>
                )
              })}
            </>
          ) : (
            <Text color={colors.dimText}>
              Enter a model id manually for this provider.
            </Text>
          )}
        </>
      )}

      {!statusBusy && step === 'confirm' && (
        <>
          <Text color={colors.text}>Confirm provider update</Text>
          <Text color={colors.muted}>
            Provider {symbols.separator} {providerId}
          </Text>
          <Text color={colors.muted}>
            Type {symbols.separator} {preset?.label ?? 'unknown'}
          </Text>
          {preset?.type === 'ollama' || preset?.type === 'custom' ? (
            <Text color={colors.muted}>
              Base URL {symbols.separator} {baseUrl}
            </Text>
          ) : (
            <Text color={colors.muted}>
              API key env {symbols.separator} {apiKeyEnvVar}
            </Text>
          )}
          {preset?.type !== 'ollama' ? (
            <Text color={colors.muted}>
              API key source {symbols.separator} {apiKeyValue.length > 0
                ? `save ${apiKeyEnvVar || 'configured env var'} to daemon .env`
                : apiKeyEnvVar
                  ? `expect ${apiKeyEnvVar} from existing shell/env`
                  : 'none (local placeholder)'}
            </Text>
          ) : null}
          {preset?.type === 'custom' ? (
            <Text color={colors.muted}>
              Custom headers {symbols.separator} {headersText.trim() ? 'configured' : 'none'}
            </Text>
          ) : null}
          <Text color={colors.muted}>
            Default model {symbols.separator} {model}
          </Text>
          <Text color={applyToCurrentSession ? colors.info : colors.warning}>
            {applyToCurrentSession
              ? 'Current session will switch to this provider after save.'
              : 'Current session will stay on its current provider after save.'}
          </Text>
          <Text color={colors.dimText}>
            This tests the provider connection first, then saves it in daemon config.
          </Text>
          <Text color={colors.dimText}>
            Use Backspace here if you need to change the provider id, authentication, headers, or model before saving.
          </Text>
          <Box gap={1} minWidth={0} width="100%">
            <Text
              color={selectedConfirmAction === 'back' ? colors.text : colors.dimText}
              backgroundColor={selectedConfirmAction === 'back' ? colors.primary : undefined}
              bold={selectedConfirmAction === 'back'}
            >
              {selectedConfirmAction === 'back' ? '>' : ' '} Back{' '}
            </Text>
            <Text
              color={selectedConfirmAction === 'save' ? colors.text : colors.success}
              backgroundColor={selectedConfirmAction === 'save' ? colors.primary : undefined}
              bold={selectedConfirmAction === 'save'}
            >
              {selectedConfirmAction === 'save' ? '>' : ' '} Test + Save{' '}
            </Text>
          </Box>
        </>
      )}

      <Text color={colors.dimText}>
        {footerHint}
      </Text>
    </Box>
  )
}
