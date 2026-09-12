import type { Timestamp } from '../types/common.js'

export interface SyncResult { direction: 'push' | 'pull' | 'both'; added: number; modified: number; deleted: number; conflicts: string[]; timestamp: Timestamp }
export interface KnowledgeSearchOptions { type?: 'semantic' | 'keyword' | 'hybrid'; limit?: number; minScore?: number; fileTypes?: string[] }
export interface KnowledgeDocument { path: string; title?: string; content: string; updatedAt: Timestamp; score?: number }

export interface IKnowledgeService {
  sync(direction: 'push' | 'pull' | 'both'): Promise<SyncResult>
  search(query: string, options?: KnowledgeSearchOptions): Promise<KnowledgeDocument[]>
  getDocument(path: string): Promise<KnowledgeDocument>
  updateDocument(path: string, content: string): Promise<KnowledgeDocument>
}
