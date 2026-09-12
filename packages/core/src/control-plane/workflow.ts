import type { Timestamp } from '../types/common.js'

export interface Board {
  id: string
  name: string
  description?: string
  columns: Column[]
  createdAt: Timestamp
}

export interface Column { id: string; name: string; position: number }

export interface Card {
  id: string; boardId: string; columnId: string; title: string; body?: string
  ticketId?: string; assignees: string[]; fields?: Record<string, unknown>
  createdAt: Timestamp; updatedAt: Timestamp
}

export interface CreateCardInput {
  title: string; body?: string; columnId: string; ticketId?: string
  assignees?: string[]; fields?: Record<string, unknown>
}

export interface CardFilter { columnId?: string; assignee?: string; ticketId?: string }
export interface CardUpdate { title?: string; body?: string; assignees?: string[]; fields?: Record<string, unknown> }

export interface IWorkflowService {
  getBoard(boardId: string): Promise<Board>
  listCards(boardId: string, filter?: CardFilter): Promise<Card[]>
  createCard(boardId: string, card: CreateCardInput): Promise<Card>
  moveCard(cardId: string, columnId: string): Promise<Card>
  updateCard(cardId: string, update: CardUpdate): Promise<Card>
}
