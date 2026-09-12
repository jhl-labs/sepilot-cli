import type { Timestamp } from '../types/common.js'
import type { PaginationParams, PaginatedResult } from '../types/pagination.js'
import type { Disposable } from '../disposable.js'

export type TicketStatus = 'open' | 'in_progress' | 'review' | 'done' | 'closed'

export interface CreateTicketInput {
  title: string
  body: string
  labels?: string[]
  assignees?: string[]
  priority?: 'critical' | 'high' | 'medium' | 'low'
  metadata?: Record<string, unknown>
}

export interface Ticket {
  id: string
  externalId?: string
  title: string
  body: string
  status: TicketStatus
  labels: string[]
  assignees: string[]
  priority: 'critical' | 'high' | 'medium' | 'low'
  createdAt: Timestamp
  updatedAt: Timestamp
  url?: string
  metadata?: Record<string, unknown>
}

export interface TicketFilter {
  status?: TicketStatus[]
  labels?: string[]
  assignee?: string
  query?: string
  pagination?: PaginationParams
}

export interface TicketUpdate {
  title?: string
  body?: string
  status?: TicketStatus
  labels?: string[]
  assignees?: string[]
  priority?: 'critical' | 'high' | 'medium' | 'low'
  metadata?: Record<string, unknown>
}

export interface CommentInput {
  body: string
  type?: 'progress' | 'result' | 'error' | 'general'
}

export interface Comment {
  id: string
  ticketId: string
  body: string
  type: 'progress' | 'result' | 'error' | 'general'
  author: string
  createdAt: Timestamp
}

export interface TicketEvent {
  id?: string
  type: 'created' | 'updated' | 'commented' | 'closed' | 'reopened' | 'labeled'
  ticket: Ticket
  changes?: Record<string, { from: unknown; to: unknown }>
  timestamp: Timestamp
}

export interface ITicketService {
  createTicket(ticket: CreateTicketInput): Promise<Ticket>
  getTicket(id: string): Promise<Ticket>
  listTickets(filter: TicketFilter): Promise<PaginatedResult<Ticket>>
  updateTicket(id: string, update: TicketUpdate): Promise<Ticket>
  addComment(ticketId: string, comment: CommentInput): Promise<Comment>
  getComments(ticketId: string): Promise<Comment[]>
  watchTickets(filter: TicketFilter, callback: (event: TicketEvent) => void): Disposable
}
