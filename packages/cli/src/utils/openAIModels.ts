/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';

export interface OpenAIModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export async function fetchOpenAIModels(apiKey: string, baseURL?: string): Promise<OpenAIModel[]> {
  try {
    const openai = new OpenAI({
      apiKey,
      baseURL,
    });

    const response = await openai.models.list();
    const models = [];
    
    for await (const model of response) {
      models.push(model);
    }

    // Filter to only chat models (exclude embedding, whisper, dall-e, etc.)
    const chatModels = models.filter(model => {
      const id = model.id.toLowerCase();
      return (
        id.includes('gpt') ||
        id.includes('claude') ||
        id.includes('llama') ||
        id.includes('mistral') ||
        id.includes('mixtral') ||
        id.includes('gemma') ||
        id.includes('qwen') ||
        id.includes('deepseek') ||
        id.includes('yi') ||
        // Common patterns for chat models
        !id.includes('embed') &&
        !id.includes('whisper') &&
        !id.includes('dall-e') &&
        !id.includes('tts') &&
        !id.includes('moderation')
      );
    });

    // Sort models by name
    return chatModels.sort((a, b) => a.id.localeCompare(b.id));
  } catch (error) {
    console.error('Failed to fetch OpenAI models:', error);
    // Return default models if API call fails
    return [
      { id: 'gpt-4o', object: 'model', created: 0, owned_by: 'openai' },
      { id: 'gpt-4-turbo-preview', object: 'model', created: 0, owned_by: 'openai' },
      { id: 'gpt-4', object: 'model', created: 0, owned_by: 'openai' },
      { id: 'gpt-3.5-turbo', object: 'model', created: 0, owned_by: 'openai' },
    ];
  }
}

export function getDefaultOpenAIModel(): string {
  return process.env.OPENAI_MODEL || 'gpt-4o';
}