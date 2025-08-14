/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { Colors } from '../colors.js';
import { RadioButtonSelect } from './shared/RadioButtonSelect.js';
import { useKeypress } from '../hooks/useKeypress.js';
import { fetchOpenAIModels, OpenAIModel, getDefaultOpenAIModel } from '../../utils/openAIModels.js';

interface ModelSelectionDialogProps {
  onSelect: (model: string) => void;
  apiKey: string;
  baseURL?: string;
}

export function ModelSelectionDialog({
  onSelect,
  apiKey,
  baseURL,
}: ModelSelectionDialogProps): React.JSX.Element {
  const [models, setModels] = useState<OpenAIModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const defaultModel = getDefaultOpenAIModel();

  useEffect(() => {
    const loadModels = async () => {
      try {
        setLoading(true);
        setError(null);
        const fetchedModels = await fetchOpenAIModels(apiKey, baseURL);
        setModels(fetchedModels);
      } catch (err) {
        setError('Failed to fetch models. Using defaults.');
        // Use default models on error
        setModels([
          { id: 'gpt-4o', object: 'model', created: 0, owned_by: 'openai' },
          { id: 'gpt-4-turbo-preview', object: 'model', created: 0, owned_by: 'openai' },
          { id: 'gpt-4', object: 'model', created: 0, owned_by: 'openai' },
          { id: 'gpt-3.5-turbo', object: 'model', created: 0, owned_by: 'openai' },
        ]);
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, [apiKey, baseURL]);

  useKeypress(
    (key) => {
      if (key.name === 'escape' && !loading) {
        // Use default model on escape
        onSelect(defaultModel);
      }
    },
    { isActive: true },
  );

  if (loading) {
    return (
      <Box
        borderStyle="round"
        borderColor={Colors.Gray}
        flexDirection="column"
        padding={1}
        width="100%"
      >
        <Text bold>Select OpenAI Model</Text>
        <Box marginTop={1}>
          <Text>Fetching available models...</Text>
        </Box>
      </Box>
    );
  }

  const items = models.map((model) => ({
    label: model.id,
    value: model.id,
  }));

  // Find default model index
  const defaultIndex = Math.max(
    0,
    items.findIndex((item) => item.value === defaultModel)
  );

  return (
    <Box
      borderStyle="round"
      borderColor={Colors.Gray}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold>Select OpenAI Model</Text>
      {error && (
        <Box marginTop={1}>
          <Text color={Colors.AccentYellow}>{error}</Text>
        </Box>
      )}
      {defaultModel !== 'gpt-4o' && (
        <Box marginTop={1}>
          <Text color={Colors.Gray}>
            Default model from OPENAI_MODEL: {defaultModel}
          </Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text>Choose a model to use:</Text>
      </Box>
      <Box marginTop={1}>
        <RadioButtonSelect
          items={items}
          initialIndex={defaultIndex}
          onSelect={onSelect}
          isFocused={true}
        />
      </Box>
      <Box marginTop={1}>
        <Text color={Colors.Gray}>
          (Use Enter to select, Escape to use default)
        </Text>
      </Box>
    </Box>
  );
}