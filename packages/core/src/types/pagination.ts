export interface PaginationParams {
  page?: number
  perPage?: number
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}

export interface PaginatedResult<T> {
  items: T[]
  totalCount: number
  page: number
  perPage: number
  hasNextPage: boolean
}
