import type { Timestamp } from '../types/common.js'

export type ChannelType = 'github-issue' | 'telegram' | 'slack' | 'discord' | 'mattermost' | 'webhook' | 'webchat' | 'whatsapp' | 'teams' | 'line'

export interface ChannelSender {
  id: string
  name: string
  type: 'user' | 'bot'
}

export interface ChannelTarget {
  id: string
  type: 'user' | 'group' | 'channel'
}

export interface ChannelAttachment {
  type: 'image' | 'file' | 'audio' | 'video'
  name: string
  mimeType: string
  data: string
  dataType: 'base64' | 'url'
}

export interface ChannelMessage {
  text: string
  attachments?: ChannelAttachment[]
  format?: 'text' | 'markdown' | 'html'
  replyTo?: string
}

export interface ChannelActivity {
  type: 'typing'
}

export interface IncomingMessage {
  channelType: ChannelType
  channelId: string
  messageId: string
  sender: ChannelSender
  text: string
  attachments?: ChannelAttachment[]
  replyTo?: string
  timestamp: Timestamp
  raw?: Record<string, unknown>
}

export type ChannelStatus = 'connected' | 'disconnected' | 'connecting' | 'error'
