/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { encode } from 'gpt-tokenizer';

export class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;
  private model: string;
  private embeddingModel: string;

  constructor(
    apiKey: string,
    model: string,
    embeddingModel: string = 'text-embedding-3-small',
    baseURL?: string,
  ) {
    this.openai = new OpenAI({
      apiKey,
      baseURL,
    });
    this.model = model;
    this.embeddingModel = embeddingModel;
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const contents = this.normalizeContents(request.contents);
    const config = (request as any).config || {};
    
    // Handle JSON response mode for OpenAI
    let messages = this.convertToOpenAIMessages(contents);
    let responseFormat: any = undefined;
    
    if (config.responseMimeType === 'application/json' && config.responseJsonSchema) {
      // Add instruction to return JSON in the specified format
      const systemMessage = {
        role: 'system' as const,
        content: `You must respond with valid JSON that conforms to this schema: ${JSON.stringify(config.responseJsonSchema)}. Return ONLY the JSON object, no additional text.`
      };
      messages = [systemMessage, ...messages];
      
      // Use OpenAI's JSON mode if available
      responseFormat = { type: 'json_object' };
    }
    
    // Add system instruction if provided
    if (config.systemInstruction) {
      const systemMessage = {
        role: 'system' as const,
        content: config.systemInstruction
      };
      messages = [systemMessage, ...messages];
    }
    
    const tools = this.convertToOpenAITools((request as any).tools);

    const completion = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      tools: tools.length > 0 ? tools : undefined,
      temperature: config.temperature ?? (request as any).generationConfig?.temperature ?? 0,
      top_p: config.topP ?? (request as any).generationConfig?.topP ?? 1,
      max_tokens: config.maxOutputTokens ?? (request as any).generationConfig?.maxOutputTokens,
      response_format: responseFormat,
    });

    return this.convertToGeminiResponse(completion);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const contents = this.normalizeContents(request.contents);
    const config = (request as any).config || {};
    
    // Handle JSON response mode for OpenAI
    let messages = this.convertToOpenAIMessages(contents);
    let responseFormat: any = undefined;
    
    if (config.responseMimeType === 'application/json' && config.responseJsonSchema) {
      // Add instruction to return JSON in the specified format
      const systemMessage = {
        role: 'system' as const,
        content: `You must respond with valid JSON that conforms to this schema: ${JSON.stringify(config.responseJsonSchema)}. Return ONLY the JSON object, no additional text.`
      };
      messages = [systemMessage, ...messages];
      
      // Use OpenAI's JSON mode if available
      responseFormat = { type: 'json_object' };
    }
    
    // Add system instruction if provided
    if (config.systemInstruction) {
      const systemMessage = {
        role: 'system' as const,
        content: config.systemInstruction
      };
      messages = [systemMessage, ...messages];
    }
    
    const tools = this.convertToOpenAITools((request as any).tools);

    const stream = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      tools: tools.length > 0 ? tools : undefined,
      temperature: config.temperature ?? (request as any).generationConfig?.temperature ?? 0,
      top_p: config.topP ?? (request as any).generationConfig?.topP ?? 1,
      max_tokens: config.maxOutputTokens ?? (request as any).generationConfig?.maxOutputTokens,
      response_format: responseFormat,
      stream: true,
    });

    const self = this;
    return (async function* () {
      for await (const chunk of stream) {
        yield self.convertStreamChunkToGeminiResponse(chunk);
      }
    })();
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Use gpt-tokenizer for token counting
    let totalTokens = 0;

    const contents = this.normalizeContents(request.contents);
    for (const content of contents) {
      if (content.parts) {
        for (const part of content.parts) {
          if (typeof part === 'string') {
            totalTokens += encode(part).length;
          } else if (part && typeof part === 'object' && 'text' in part) {
            const textPart = part as any;
            if (textPart.text) {
              totalTokens += encode(textPart.text).length;
            }
          }
        }
      }
    }

    return {
      totalTokens,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    const text = this.extractTextFromContent((request as any).contents || (request as any).content);
    
    const response = await this.openai.embeddings.create({
      model: this.embeddingModel,
      input: text,
    });

    return {
      embeddings: [{
        values: response.data[0].embedding,
      }],
    };
  }

  private normalizeContents(contents: any): Content[] {
    if (Array.isArray(contents)) {
      // Check if it's an array of Content objects
      if (contents.length === 0 || (contents[0] && typeof contents[0] === 'object' && 'role' in contents[0])) {
        return contents as Content[];
      }
      // If it's an array of parts, wrap in a user message
      return [{ role: 'user', parts: contents }];
    }
    // If it's a single string or part, wrap in a user message
    if (typeof contents === 'string') {
      return [{ role: 'user', parts: [{ text: contents }] }];
    }
    // Otherwise assume it's a single Content or Part
    return [{ role: 'user', parts: [contents] }];
  }

  private convertToOpenAIMessages(contents: Content[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    for (const content of contents) {
      if (content.role === 'user') {
        const messageContent = this.convertPartsToOpenAI(content.parts || []);
        messages.push({
          role: 'user',
          content: messageContent,
        });
      } else if (content.role === 'model') {
        const assistantMessage: OpenAI.Chat.ChatCompletionAssistantMessageParam = {
          role: 'assistant',
          content: null,
        };

        if (content.parts) {
          const textParts: string[] = [];
          const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = [];

          for (const part of content.parts) {
            if (typeof part === 'string') {
              textParts.push(part);
            } else if (part && typeof part === 'object') {
              if ('text' in part) {
                textParts.push(part.text || '');
              } else if ('functionCall' in part && part.functionCall) {
                toolCalls.push({
                  id: `call_${Math.random().toString(36).substr(2, 9)}`,
                  type: 'function',
                  function: {
                    name: part.functionCall.name || '',
                    arguments: JSON.stringify(part.functionCall.args),
                  },
                });
              }
            }
          }

          if (textParts.length > 0) {
            assistantMessage.content = textParts.join('\n');
          } else if (toolCalls.length === 0) {
            assistantMessage.content = '';
          }
          if (toolCalls.length > 0) {
            assistantMessage.tool_calls = toolCalls;
          }
        }

        messages.push(assistantMessage);
      } else if (content.role === 'function') {
        // Handle function responses
        if (content.parts) {
          for (const part of content.parts) {
            if (part && typeof part === 'object' && 'functionResponse' in part && part.functionResponse) {
              messages.push({
                role: 'tool',
                content: JSON.stringify(part.functionResponse.response),
                tool_call_id: part.functionResponse.name || '',
              });
            }
          }
        }
      }
    }

    return messages;
  }

  private convertPartsToOpenAI(parts: Part[]): string | OpenAI.Chat.ChatCompletionContentPart[] {
    const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [];
    
    for (const part of parts) {
      if (typeof part === 'string') {
        contentParts.push({ type: 'text', text: part });
      } else if (part && typeof part === 'object') {
        if ('text' in part) {
          contentParts.push({ type: 'text', text: part.text || '' });
        } else if ('inlineData' in part && part.inlineData) {
          contentParts.push({
            type: 'image_url',
            image_url: {
              url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`,
            },
          });
        }
      }
    }

    // If all parts are text, return as a single string
    if (contentParts.every(p => p.type === 'text')) {
      return contentParts.map(p => (p as any).text).join('\n');
    }

    return contentParts;
  }

  private convertToOpenAITools(tools?: any[]): OpenAI.Chat.ChatCompletionTool[] {
    if (!tools || tools.length === 0) return [];

    const openAITools: OpenAI.Chat.ChatCompletionTool[] = [];

    for (const tool of tools) {
      if (tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          openAITools.push({
            type: 'function',
            function: {
              name: func.name,
              description: func.description,
              parameters: func.parameters || {},
            },
          });
        }
      }
    }

    return openAITools;
  }

  private convertToGeminiResponse(completion: OpenAI.Chat.ChatCompletion): GenerateContentResponse {
    const choice = completion.choices[0];
    const parts: Part[] = [];

    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        if ('function' in toolCall) {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments || '{}'),
            },
          });
        }
      }
    }

    // Extract text content for the response
    let textContent = '';
    for (const part of parts) {
      if (part && typeof part === 'object' && 'text' in part && part.text) {
        textContent = part.text;
        break;
      }
    }
    
    const response: GenerateContentResponse = {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          index: 0,
        },
      ],
      usageMetadata: {
        promptTokenCount: completion.usage?.prompt_tokens || 0,
        candidatesTokenCount: completion.usage?.completion_tokens || 0,
        totalTokenCount: completion.usage?.total_tokens || 0,
      },
      // Add text helper for compatibility with Gemini's response format
      text: textContent,
    } as GenerateContentResponse;
    
    // Debug logging for empty responses
    if (!textContent && process.env.DEBUG) {
      console.error('OpenAI returned empty text. Parts:', JSON.stringify(parts));
    }

    return response;
  }

  private convertStreamChunkToGeminiResponse(chunk: OpenAI.Chat.ChatCompletionChunk): GenerateContentResponse {
    const choice = chunk.choices[0];
    const parts: Part[] = [];

    // Debug logging for streaming issues
    if (process.env.DEBUG_OPENAI) {
      console.log('OpenAI chunk:', JSON.stringify(choice?.delta));
    }

    if (choice?.delta?.content) {
      parts.push({ text: choice.delta.content });
    }

    if (choice?.delta?.tool_calls) {
      for (const toolCall of choice.delta.tool_calls) {
        if (toolCall.function?.name) {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: toolCall.function.arguments ? JSON.parse(toolCall.function.arguments) : {},
            },
          });
        }
      }
    }

    // Extract text content for the response
    let textContent = '';
    for (const part of parts) {
      if (part && typeof part === 'object' && 'text' in part && part.text) {
        textContent = part.text;
        break;
      }
    }
    
    const response: GenerateContentResponse = {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          index: 0,
        },
      ],
      // Add text helper for compatibility with Gemini's response format
      text: textContent,
    } as GenerateContentResponse;

    return response;
  }

  private extractTextFromContent(content: Content | Part | string): string {
    if (typeof content === 'string') {
      return content;
    }

    if ('parts' in content && content.parts) {
      return content.parts
        .map(part => this.extractTextFromContent(part))
        .filter(Boolean)
        .join(' ');
    }

    if ('text' in content) {
      const textPart = content as any;
      return textPart.text || '';
    }

    return '';
  }
}