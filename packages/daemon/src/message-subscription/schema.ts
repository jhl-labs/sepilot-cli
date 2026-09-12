import { z } from 'zod'

export const MessageSubscriptionConfigInput = z.object({
  enabled: z.boolean().default(false),
  connectionType: z.enum(['polling', 'websocket', 'nats']).default('polling'),
  pollingUrl: z.string().default(''),
  websocketUrl: z.string().default(''),
  pollingInterval: z.number().int().min(1000).default(60_000),
  authToken: z.string().default(''),
  customHeaders: z.record(z.string(), z.string()).default({}),
  natsUrl: z.string().default(''),
  natsConsumerId: z.string().default(''),
  natsConsumerSecret: z.string().default(''),
  natsStreamName: z.string().default('WEBHOOKS'),
  natsSubject: z.string().default('webhooks.>'),
  natsBatchSize: z.number().int().min(1).default(10),
  natsFetchTimeout: z.number().int().min(1000).default(5000),
  maxQueueSize: z.number().int().min(1).default(1000),
  retentionDays: z.number().int().min(1).default(7),
  autoProcess: z.boolean().default(true),
  retryAttempts: z.number().int().min(0).default(3),
  retryDelay: z.number().int().min(0).default(5000),
  useAIProcessing: z.boolean().default(false),
  aiPromptTemplate: z.string().default(''),
  thinkingMode: z.enum(['instant', 'sequential']).default('instant'),
  showNotification: z.boolean().default(true),
})

export type MessageSubscriptionConfig = z.infer<
  typeof MessageSubscriptionConfigInput
>

export const DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG =
  MessageSubscriptionConfigInput.parse({})

export type MessageStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface MessageSubscriptionItem {
  hash: string
  id: string | null
  type: 'github_webhook' | 'community_post' | 'custom'
  source: string
  title: string
  body: string
  content: string
  metadata: Record<string, unknown>
  timestamp: number
  queuedAt: number
  status: MessageStatus
  processedAt: number | null
  error: string | null
  retryCount: number
  conversationId: string | null
}

export interface MessageQueueStatus {
  pending: number
  processing: number
  completed: number
  failed: number
  totalProcessed: number
  lastPolled: number | null
  lastProcessed: number | null
}

export interface SubscriptionStatus {
  isConnected: boolean
  lastPolled: number | null
  lastError: string | null
}
