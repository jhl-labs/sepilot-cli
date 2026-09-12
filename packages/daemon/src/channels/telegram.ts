import TelegramBot from 'node-telegram-bot-api'
import { randomInt } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable, ChannelActivity } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import { splitChannelMessage } from './message-chunker.js'
import { UNSUPPORTED_MEDIA_MESSAGE } from './media-ack.js'

const log = createLogger('channel:telegram')
const TELEGRAM_MESSAGE_CHAR_LIMIT = 3900
const TELEGRAM_BOT_COMMANDS = [
  { command: 'start', description: 'Show pairing instructions' },
  { command: 'help', description: 'Show channel commands' },
  { command: 'health', description: 'Show daemon health' },
  { command: 'session', description: 'Show the active chat session' },
  { command: 'new', description: 'Start a fresh chat session' },
  { command: 'sessions', description: 'List recent sessions' },
  { command: 'resume', description: 'Resume a previous session' },
  { command: 'model', description: 'Show or switch the active model' },
  { command: 'schedule', description: 'List and manage scheduled tasks' },
  { command: 'memory', description: 'Show durable memory status' },
  { command: 'skills', description: 'List, search, install, or toggle skills' },
  { command: 'tps', description: 'Show token output speed stats' },
  { command: 'usage', description: 'Alias for token output speed stats' },
  { command: 'stop', description: 'Stop active work in this chat' },
  { command: 'cwd', description: 'Show or change the workspace' },
]
const SCHEDULE_CALLBACK_PREFIX = 'sch'
const SCHEDULE_LIST_ITEM_PATTERN = /^\d+\.\s+([A-Za-z0-9_-]{4,64})\s+·\s+[^·]+·\s+([^·\n]+)\s+·/gm
const SCHEDULE_CALLBACK_PATTERN = /^sch:(show|runs|pause|resume|run|cancel):([A-Za-z0-9_-]{4,64})$/

type TelegramInlineKeyboard = {
  inline_keyboard: Array<Array<{ text: string; callback_data: string }>>
}

type TelegramAttachmentSource = Buffer | string

type TelegramAttachmentBot = {
  sendPhoto?: (
    chatId: number,
    photo: TelegramAttachmentSource,
    options?: Record<string, unknown>,
    fileOptions?: Record<string, unknown>,
  ) => Promise<unknown>
  sendDocument?: (
    chatId: number,
    document: TelegramAttachmentSource,
    options?: Record<string, unknown>,
    fileOptions?: Record<string, unknown>,
  ) => Promise<unknown>
  sendAudio?: (
    chatId: number,
    audio: TelegramAttachmentSource,
    options?: Record<string, unknown>,
    fileOptions?: Record<string, unknown>,
  ) => Promise<unknown>
  sendVideo?: (
    chatId: number,
    video: TelegramAttachmentSource,
    options?: Record<string, unknown>,
    fileOptions?: Record<string, unknown>,
  ) => Promise<unknown>
}

function parseReplyToMessageId(replyTo: string | undefined): number | undefined {
  if (!replyTo) return undefined
  const parsed = parseInt(replyTo, 10)
  return Number.isFinite(parsed) ? parsed : undefined
}

function telegramAttachmentSource(
  attachment: NonNullable<ChannelMessage['attachments']>[number],
): TelegramAttachmentSource {
  if (attachment.dataType === 'base64') return Buffer.from(attachment.data, 'base64')
  return attachment.data
}

function buildScheduleInlineKeyboard(text: string): TelegramInlineKeyboard | undefined {
  if (!text.startsWith('📅 예약된 작업')) return undefined

  SCHEDULE_LIST_ITEM_PATTERN.lastIndex = 0
  const rows: TelegramInlineKeyboard['inline_keyboard'] = []
  let match: RegExpExecArray | null
  let count = 0
  while ((match = SCHEDULE_LIST_ITEM_PATTERN.exec(text)) !== null && count < 8) {
    const id = match[1]
    const status = match[2]?.trim().toLowerCase()
    const toggleAction = status === 'paused' || status === '일시정지' ? 'resume' : 'pause'
    rows.push([
      { text: `${id} 보기`, callback_data: `${SCHEDULE_CALLBACK_PREFIX}:show:${id}` },
      { text: '기록', callback_data: `${SCHEDULE_CALLBACK_PREFIX}:runs:${id}` },
    ])
    rows.push([
      {
        text: toggleAction === 'resume' ? '재개' : '정지',
        callback_data: `${SCHEDULE_CALLBACK_PREFIX}:${toggleAction}:${id}`,
      },
      { text: '실행', callback_data: `${SCHEDULE_CALLBACK_PREFIX}:run:${id}` },
      { text: '취소', callback_data: `${SCHEDULE_CALLBACK_PREFIX}:cancel:${id}` },
    ])
    count += 1
  }

  return rows.length > 0 ? { inline_keyboard: rows } : undefined
}

function scheduleCallbackToSlash(data: string | undefined): string | null {
  if (!data) return null
  const match = SCHEDULE_CALLBACK_PATTERN.exec(data)
  if (!match) return null
  return `/schedule ${match[1]} ${match[2]}`
}

export function splitTelegramMessage(text: string): string[] {
  return splitChannelMessage(text, { limit: TELEGRAM_MESSAGE_CHAR_LIMIT })
}

export interface TelegramPendingPairing {
  code: string
  expiresAt: number
}

export interface TelegramChannelConfig {
  botToken: string
  allowedUsers: string[]
  pairingRequired?: boolean
  pairingCodeTtl?: number  // seconds, default 300
  rateLimitPerMinute?: number  // default 30
  onPairedUser?: (userId: string) => Promise<void> | void
  loadPendingPairing?: () => Promise<TelegramPendingPairing | null> | TelegramPendingPairing | null
  clearPendingPairing?: () => Promise<void> | void
  /**
   * Optional voice-note transcriber. When wired (by the runtime, to a
   * local whisper binary — never an external API), an incoming voice /
   * audio message is downloaded, transcribed, and the transcript is
   * dispatched into the pipeline as if the user had typed it. Returns
   * the transcript text; throws if transcription is unavailable so the
   * channel can fall back to a "please type" reply. Leave unset to
   * politely decline voice messages.
   */
  transcribeAudioFile?: (audioPath: string) => Promise<string>
  /**
   * Optional reply synthesizer. When wired (by the runtime, to a local
   * piper binary — never an external API), and the most recent inbound
   * in a chat was a voice note, the agent's reply is *also* sent as a
   * voice note (modality mirroring). Returns the path to a WAV plus a
   * `temporary` flag the channel uses to clean up after sending.
   */
  synthesizeReply?: (text: string) => Promise<{ audioPath: string; temporary: boolean }>
  /**
   * Maximum inbound voice/audio size (bytes) the channel will download for
   * transcription. A paired user could otherwise force unbounded downloads of
   * large audio files, pressuring disk. Default 20 MiB.
   */
  maxVoiceDownloadBytes?: number
}

/** Default cap on inbound voice/audio downloads (bytes). */
const DEFAULT_MAX_VOICE_DOWNLOAD_BYTES = 20 * 1024 * 1024

/** A chat's "last inbound was a voice note" flag is treated as stale after this. */
const VOICE_REPLY_WINDOW_MS = 5 * 60_000
/** Don't voice replies longer than this — a multi-minute spoken code dump is useless. */
const VOICE_REPLY_MAX_CHARS = 4_000

export interface TelegramPairingCode {
  code: string
  expiresAt: string
}

export class TelegramChannel implements IChannel {
  readonly id = 'telegram'
  readonly type: ChannelType = 'telegram'
  private bot: TelegramBot | null = null
  private config: TelegramChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []
  private allowedUsers: Set<string>
  private pendingPairing: TelegramPendingPairing | null = null
  private messageCount = new Map<string, { count: number; resetAt: number }>()
  /** chatId → ms timestamp of the most recent transcribed voice-note inbound. */
  private voiceReplyChats = new Map<string, number>()

  constructor(config: TelegramChannelConfig) {
    this.config = config
    this.allowedUsers = new Set(config.allowedUsers ?? [])
  }

  async start(): Promise<void> {
    this.status = 'connecting'
    try {
      this.bot = new TelegramBot(this.config.botToken, { polling: true })

      this.bot.on('message', (msg) => this.handleMessage(msg))
      this.bot.on('callback_query', (query) => this.handleCallbackQuery(query))
      await this.registerBotCommands()

      this.status = 'connected'
    } catch (err) {
      this.status = 'error'
      throw err
    }
  }

  async stop(): Promise<void> {
    if (this.bot) {
      await this.bot.stopPolling()
      this.bot = null
    }
    this.status = 'disconnected'
  }

  private async loadPendingPairing(): Promise<TelegramPendingPairing | null> {
    if (this.pendingPairing) {
      return this.pendingPairing
    }

    try {
      const pairing = await this.config.loadPendingPairing?.()
      this.pendingPairing = pairing ?? null
      return this.pendingPairing
    } catch (error) {
      log.warn(`Failed to load pending pairing: ${error}`)
      return null
    }
  }

  private async clearPendingPairing(): Promise<void> {
    this.pendingPairing = null
    try {
      await this.config.clearPendingPairing?.()
    } catch (error) {
      log.warn(`Failed to clear pending pairing: ${error}`)
    }
  }

  getStatus(): ChannelStatus {
    return this.status
  }

  onMessage(handler: (msg: IncomingMessage) => Promise<void>): Disposable {
    this.handlers.push(handler)
    return {
      dispose: () => {
        const idx = this.handlers.indexOf(handler)
        if (idx >= 0) this.handlers.splice(idx, 1)
      },
    }
  }

  private async registerBotCommands(): Promise<void> {
    const bot = this.bot as unknown as {
      setMyCommands?: (commands: Array<{ command: string; description: string }>) => Promise<unknown> | unknown
    } | null
    if (!bot?.setMyCommands) return
    try {
      await bot.setMyCommands(TELEGRAM_BOT_COMMANDS)
    } catch (error) {
      log.warn(`Failed to register Telegram bot commands: ${error}`)
    }
  }

  async sendMessage(target: ChannelTarget, msg: ChannelMessage): Promise<void> {
    if (!this.bot) throw new Error('Telegram bot not started')
    const chatId = parseInt(target.id)
    const parseMode = msg.format === 'markdown' ? 'Markdown' : undefined
    const replyToMessageId = parseReplyToMessageId(msg.replyTo)

    let isFirstChunk = true
    for (const chunk of splitTelegramMessage(msg.text)) {
      const replyOptions = isFirstChunk && replyToMessageId !== undefined
        ? { reply_to_message_id: replyToMessageId, allow_sending_without_reply: true }
        : {}
      const replyMarkup = isFirstChunk ? buildScheduleInlineKeyboard(chunk) : undefined
      const keyboardOptions = replyMarkup ? { reply_markup: replyMarkup } : {}
      isFirstChunk = false

      if (!parseMode) {
        await this.bot.sendMessage(chatId, chunk, { parse_mode: undefined, ...replyOptions, ...keyboardOptions })
        continue
      }

      try {
        await this.bot.sendMessage(chatId, chunk, { parse_mode: parseMode, ...replyOptions, ...keyboardOptions })
      } catch (error) {
        log.warn(`Telegram markdown delivery failed; retrying as plain text: ${error}`)
        await this.bot.sendMessage(chatId, chunk, { parse_mode: undefined, ...replyOptions, ...keyboardOptions })
      }
    }

    await this.sendAttachments(chatId, msg, replyToMessageId)

    // Modality mirroring: if the most recent inbound in this chat was a
    // voice note and a TTS synthesizer is wired, also deliver the reply
    // as a voice note. Best-effort — the text was already sent, so a
    // synthesis failure is logged and swallowed.
    await this.maybeSendVoiceReply(target.id, chatId, msg.text)
  }

  private async sendAttachments(
    chatId: number,
    msg: ChannelMessage,
    replyToMessageId: number | undefined,
  ): Promise<void> {
    if (!msg.attachments?.length || !this.bot) return
    const bot = this.bot as unknown as TelegramAttachmentBot
    const replyOptions = replyToMessageId !== undefined
      ? {
          reply_to_message_id: replyToMessageId,
          allow_sending_without_reply: true,
        }
      : {}

    for (const attachment of msg.attachments) {
      const source = telegramAttachmentSource(attachment)
      const fileOptions = {
        filename: attachment.name,
        contentType: attachment.mimeType,
      }
      if (attachment.type === 'image') {
        if (!bot.sendPhoto) throw new Error('Telegram bot cannot send image attachments')
        await bot.sendPhoto(chatId, source, replyOptions, fileOptions)
        continue
      }
      if (attachment.type === 'audio') {
        if (!bot.sendAudio) throw new Error('Telegram bot cannot send audio attachments')
        await bot.sendAudio(chatId, source, replyOptions, fileOptions)
        continue
      }
      if (attachment.type === 'video') {
        if (!bot.sendVideo) throw new Error('Telegram bot cannot send video attachments')
        await bot.sendVideo(chatId, source, replyOptions, fileOptions)
        continue
      }
      if (!bot.sendDocument) throw new Error('Telegram bot cannot send file attachments')
      await bot.sendDocument(chatId, source, replyOptions, fileOptions)
    }
  }

  private async maybeSendVoiceReply(targetId: string, chatId: number, text: string): Promise<void> {
    if (!this.config.synthesizeReply || !this.bot) return
    const last = this.voiceReplyChats.get(targetId)
    if (last === undefined || Date.now() - last > VOICE_REPLY_WINDOW_MS) return
    const trimmed = text.trim()
    if (!trimmed || trimmed.length > VOICE_REPLY_MAX_CHARS) return

    let result: { audioPath: string; temporary: boolean } | undefined
    try {
      result = await this.config.synthesizeReply(trimmed)
      await this.bot.sendVoice(chatId, result.audioPath)
    } catch (error) {
      log.warn('Voice reply synthesis/send failed', { error: error instanceof Error ? error.message : String(error) })
    } finally {
      if (result?.temporary) {
        // synthesizeSpeech writes the WAV inside a temp dir; remove the dir.
        const { dirname } = await import('node:path')
        await rm(dirname(result.audioPath), { recursive: true, force: true }).catch(() => {})
      }
    }
  }

  private noteInboundModality(chatId: string, fromVoice: boolean): void {
    if (fromVoice) {
      // Opportunistic prune of stale entries while we're here.
      const cutoff = Date.now() - VOICE_REPLY_WINDOW_MS
      for (const [key, ts] of this.voiceReplyChats) {
        if (ts < cutoff) this.voiceReplyChats.delete(key)
      }
      this.voiceReplyChats.set(chatId, Date.now())
    } else {
      this.voiceReplyChats.delete(chatId)
    }
  }

  private async answerCallbackQuery(id: string, text?: string): Promise<void> {
    const bot = this.bot as unknown as {
      answerCallbackQuery?: (callbackQueryId: string, options?: { text?: string }) => Promise<unknown> | unknown
    } | null
    try {
      await bot?.answerCallbackQuery?.(id, text ? { text } : undefined)
    } catch (error) {
      log.warn(`Failed to answer Telegram callback query: ${error}`)
    }
  }

  private async handleCallbackQuery(query: TelegramBot.CallbackQuery): Promise<void> {
    const text = scheduleCallbackToSlash(query.data)
    if (!text) {
      await this.answerCallbackQuery(query.id, '지원하지 않는 작업입니다.')
      return
    }

    const chatId = query.message?.chat?.id
    if (chatId === undefined) {
      await this.answerCallbackQuery(query.id, '채팅 정보를 찾을 수 없습니다.')
      return
    }

    const userId = String(query.from.id)
    const channelId = String(chatId)
    if (!this.checkRateLimit(userId)) {
      await this.answerCallbackQuery(query.id)
      return
    }

    if (this.config.pairingRequired !== false && !this.allowedUsers.has(userId)) {
      await this.answerCallbackQuery(query.id, '먼저 /pair 로 연결해주세요.')
      await this.bot?.sendMessage(channelId, 'You are not paired. Use /pair <code> to connect.')
      return
    }

    await this.answerCallbackQuery(query.id)
    this.noteInboundModality(channelId, false)

    const incoming: IncomingMessage = {
      channelType: 'telegram',
      channelId,
      messageId: `callback:${query.id}`,
      sender: {
        id: userId,
        name: query.from.first_name + (query.from.last_name ? ` ${query.from.last_name}` : ''),
        type: query.from.is_bot ? 'bot' : 'user',
      },
      text,
      ...(query.message?.message_id
        ? { replyTo: String(query.message.message_id) }
        : {}),
      timestamp: new Date().toISOString(),
      raw: {
        chatType: query.message?.chat.type,
      },
    }

    for (const handler of this.handlers) {
      try {
        await handler(incoming)
      } catch (err) {
        log.error(`Handler error: ${err}`)
      }
    }
  }

  async sendActivity(target: ChannelTarget, activity: ChannelActivity): Promise<void> {
    if (!this.bot) throw new Error('Telegram bot not started')
    if (activity.type !== 'typing') return
    await this.bot.sendChatAction(parseInt(target.id), 'typing')
  }

  /** Generate a pairing code for CLI to display */
  generatePairingCode(): TelegramPairingCode {
    const code = String(randomInt(100000, 999999))
    const expiresAt = Date.now() + (this.config.pairingCodeTtl ?? 300) * 1000
    this.pendingPairing = {
      code,
      expiresAt,
    }
    return {
      code,
      expiresAt: new Date(expiresAt).toISOString(),
    }
  }

  revokeAllowedUser(userId: string): boolean {
    const removed = this.allowedUsers.delete(userId)
    if (!removed) {
      return false
    }

    this.config.allowedUsers = this.config.allowedUsers.filter(
      (value) => value !== userId,
    )
    return true
  }

  private async handleMessage(msg: TelegramBot.Message): Promise<void> {
    if (!msg.from) return

    const userId = String(msg.from.id)
    const chatId = String(msg.chat.id)

    // Rate-limit at the entrance before any outbound reply. This
    // covers /start, unpaired prompts, and pairing replies. The
    // previous flow only checked after the unpaired branch, so an
    // attacker hitting /start (or messaging from an unpaired
    // account) could force this bot into uncapped outbound traffic
    // — Telegram's send budget would burn out and the bot would be
    // rate-limited or temporarily banned.
    if (!this.checkRateLimit(userId)) {
      // Drop silently. Replying with "Rate limit exceeded" here
      // would itself amplify, defeating the purpose. The legitimate
      // path replies to messages that pass the limit; abusive
      // senders get nothing.
      return
    }

    const hasText = typeof msg.text === 'string' && msg.text.trim().length > 0
    const hasAudio = Boolean(msg.voice || msg.audio)
    if (!hasText && !hasAudio) {
      // Distinguish real media (photo/document/video/sticker/…) — which the
      // user actively sent and deserves an ack — from content-less updates
      // (empty edits, service messages) which stay silent.
      const hasMedia = Boolean(
        msg.photo || msg.document || msg.video || msg.sticker
        || msg.animation || msg.video_note || msg.location || msg.contact,
      )
      if (hasMedia && (this.config.pairingRequired === false || this.allowedUsers.has(userId))) {
        await this.bot?.sendMessage(chatId, UNSUPPORTED_MEDIA_MESSAGE)
      }
      return
    }

    // Text-only slash commands.
    if (typeof msg.text === 'string') {
      if (msg.text.startsWith('/pair ')) {
        await this.handlePairing(msg)
        return
      }
      if (msg.text === '/start') {
        await this.bot?.sendMessage(chatId, 'Welcome to sepilotd! Use /pair <code> to connect.')
        return
      }
    }

    // Access gate — covers text and voice; an unpaired sender always
    // gets the pairing hint regardless of message type.
    if (this.config.pairingRequired !== false && !this.allowedUsers.has(userId)) {
      await this.bot?.sendMessage(chatId, 'You are not paired. Use /pair <code> to connect.')
      return
    }

    // Resolve the message body: typed text, or a transcribed voice /
    // audio note. Returns null when transcription already replied with
    // an explanation (unavailable / failed).
    const text = await this.resolveMessageText(msg, chatId)
    if (text == null || text.length === 0) return

    // Track the inbound modality so a reply to a voice note can be
    // voiced back (and a later typed message switches the chat back to
    // text-only replies).
    this.noteInboundModality(chatId, !hasText)

    const incoming: IncomingMessage = {
      channelType: 'telegram',
      channelId: chatId,
      messageId: String(msg.message_id),
      sender: {
        id: userId,
        name: msg.from.first_name + (msg.from.last_name ? ` ${msg.from.last_name}` : ''),
        type: msg.from.is_bot ? 'bot' : 'user',
      },
      text,
      ...(msg.reply_to_message?.message_id
        ? { replyTo: String(msg.reply_to_message.message_id) }
        : {}),
      timestamp: new Date(msg.date * 1000).toISOString(),
      raw: {
        chatType: msg.chat.type,
      },
    }

    for (const handler of this.handlers) {
      try {
        await handler(incoming)
      } catch (err) {
        log.error(`Handler error: ${err}`)
      }
    }
  }

  private async resolveMessageText(
    msg: TelegramBot.Message,
    chatId: string,
  ): Promise<string | null> {
    if (typeof msg.text === 'string' && msg.text.trim().length > 0) {
      return msg.text
    }

    const voice = msg.voice ?? msg.audio
    if (!voice) {
      // Photos, stickers, documents, etc. are not handled here.
      return null
    }
    if (!this.config.transcribeAudioFile) {
      await this.bot?.sendMessage(
        chatId,
        '음성 메시지 처리가 설정되어 있지 않습니다. 텍스트로 보내주세요. (Voice transcription is not configured — please send text.)',
      )
      return null
    }
    if (!this.bot) return null

    // Cap inbound audio size before downloading — a paired user could
    // otherwise force unbounded large-file downloads and pressure disk.
    const maxBytes = this.config.maxVoiceDownloadBytes ?? DEFAULT_MAX_VOICE_DOWNLOAD_BYTES
    const fileSize = (voice as { file_size?: number }).file_size
    if (typeof fileSize === 'number' && fileSize > maxBytes) {
      await this.bot.sendMessage(
        chatId,
        `음성 파일이 너무 큽니다 (최대 ${Math.floor(maxBytes / (1024 * 1024))}MB). 더 짧게 보내주시거나 텍스트로 입력해주세요.`,
      )
      return null
    }

    let dir: string | undefined
    try {
      dir = await mkdtemp(join(tmpdir(), 'sepilotd-tg-voice-'))
      // node-telegram-bot-api downloads the file and returns its path.
      const localPath = await this.bot.downloadFile(voice.file_id, dir)
      const transcript = (await this.config.transcribeAudioFile(localPath)).trim()
      if (!transcript) {
        await this.bot.sendMessage(chatId, '음성에서 텍스트를 추출하지 못했습니다. 다시 보내주시거나 텍스트로 입력해주세요.')
        return null
      }
      return transcript
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      log.warn('Voice transcription failed', { error: message })
      await this.bot.sendMessage(chatId, `음성 메시지를 처리하지 못했습니다: ${message}`)
      return null
    } finally {
      if (dir) {
        await rm(dir, { recursive: true, force: true }).catch(() => {})
      }
    }
  }

  private async handlePairing(msg: TelegramBot.Message): Promise<void> {
    const chatId = String(msg.chat.id)
    const userId = String(msg.from!.id)
    const code = msg.text!.split(' ')[1]?.trim()

    const pendingPairing = await this.loadPendingPairing()
    if (!pendingPairing) {
      await this.bot?.sendMessage(chatId, 'No pairing code available. Generate one from CLI: sepilot channel pair telegram')
      return
    }

    if (Date.now() > pendingPairing.expiresAt) {
      await this.clearPendingPairing()
      await this.bot?.sendMessage(chatId, 'Pairing code expired. Generate a new one.')
      return
    }

    if (code !== pendingPairing.code) {
      await this.bot?.sendMessage(chatId, 'Invalid pairing code.')
      return
    }

    try {
      await this.config.onPairedUser?.(userId)
    } catch (error) {
      log.error(`Failed to persist pairing: ${error}`)
      await this.bot?.sendMessage(chatId, 'Pairing failed. Please try again in a moment.')
      return
    }

    this.allowedUsers.add(userId)
    if (!this.config.allowedUsers.includes(userId)) {
      this.config.allowedUsers.push(userId)
    }
    await this.clearPendingPairing()
    await this.bot?.sendMessage(chatId, 'Paired successfully! You can now send messages.')
  }

  private checkRateLimit(userId: string): boolean {
    const limit = this.config.rateLimitPerMinute ?? 30
    const now = Date.now()
    const entry = this.messageCount.get(userId)

    if (!entry || now > entry.resetAt) {
      this.messageCount.set(userId, { count: 1, resetAt: now + 60000 })
      return true
    }

    if (entry.count >= limit) return false
    entry.count++
    return true
  }
}
