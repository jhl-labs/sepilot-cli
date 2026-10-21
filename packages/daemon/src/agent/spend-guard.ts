export interface SpendBudgetConfig {
  dailyUsdBudget?: number
  sessionUsdBudget?: number
}

export interface SpendScope {
  sessionId: string
}

export interface SpendUsageSource {
  getDailySummaries: (days?: number) => Array<{ totalCostUsd: number }>
  getSessionUsage: (sessionId: string) => { costUsd: number }
}

export interface SpendCheckResult {
  allowed: boolean
  reason?: string
}

function envBudget(name: string): number | undefined {
  const raw = process.env[name]
  if (!raw) return undefined
  const value = Number(raw)
  return Number.isFinite(value) && value > 0 ? value : undefined
}

export function resolveSpendBudgets(config: SpendBudgetConfig | undefined): SpendBudgetConfig {
  return {
    dailyUsdBudget: envBudget('SEPILOTD_DAILY_USD_BUDGET') ?? config?.dailyUsdBudget,
    sessionUsdBudget: envBudget('SEPILOTD_SESSION_USD_BUDGET') ?? config?.sessionUsdBudget,
  }
}

export function checkSpendBudget(
  usage: SpendUsageSource,
  config: SpendBudgetConfig | undefined,
  scope: SpendScope,
): SpendCheckResult {
  const budgets = resolveSpendBudgets(config)
  if (budgets.dailyUsdBudget === undefined && budgets.sessionUsdBudget === undefined) {
    return { allowed: true }
  }

  if (budgets.dailyUsdBudget !== undefined) {
    const spentToday = usage
      .getDailySummaries(1)
      .reduce((sum, row) => sum + (row.totalCostUsd ?? 0), 0)
    if (spentToday >= budgets.dailyUsdBudget) {
      return {
        allowed: false,
        reason: `daily spend budget $${budgets.dailyUsdBudget} exceeded (spent $${spentToday.toFixed(4)} today)`,
      }
    }
  }

  if (budgets.sessionUsdBudget !== undefined) {
    const spentSession = usage.getSessionUsage(scope.sessionId).costUsd ?? 0
    if (spentSession >= budgets.sessionUsdBudget) {
      return {
        allowed: false,
        reason: `session spend budget $${budgets.sessionUsdBudget} exceeded (spent $${spentSession.toFixed(4)} this session)`,
      }
    }
  }

  return { allowed: true }
}
