export interface PortfolioHoldingDefinition {
  label: string
  symbol: string
  aliases: string[]
}

export interface PortfolioHolding {
  label: string
  symbol: string
  averageCost: number
  quantity: number
}

export const PORTFOLIO_HOLDING_DEFINITIONS: PortfolioHoldingDefinition[] = [
  {
    label: '삼성전자',
    symbol: '005930',
    aliases: ['삼성전자', 'samsung electronics', '005930'],
  },
  {
    label: 'SK하이닉스',
    symbol: '000660',
    aliases: ['sk하이닉스', '하이닉스', 'sk hynix', '000660'],
  },
  {
    label: 'KODEX 반도체',
    symbol: '091160',
    aliases: ['kodex반도체', 'kodex 반도체', 'kodex semiconductor', '091160'],
  },
  {
    label: 'KODEX 미국S&P500',
    symbol: '379800',
    aliases: ['kodex미국s&p500', 'kodex 미국 s&p 500', 'kodex 미국s&p500', 'kodex us s&p500', '379800'],
  },
  {
    label: 'KODEX AI반도체',
    symbol: '395160',
    aliases: ['kodexai반도체', 'kodex ai 반도체', 'kodex ai반도체', '395160'],
  },
]

interface NumericToken {
  raw: string
  value: number
  unit?: string
}

function compactWithIndexMap(text: string): { compact: string; indexMap: number[] } {
  let compact = ''
  const indexMap: number[] = []
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    if (!char || /\s/.test(char)) continue
    compact += char.toLowerCase()
    indexMap.push(index)
  }
  return { compact, indexMap }
}

function parseNumber(value: string | undefined): number | null {
  if (!value) return null
  const parsed = Number(value.replace(/,/g, ''))
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function parseNumericTokens(text: string): NumericToken[] {
  return Array.from(text.matchAll(/([+-]?\d[\d,]*)(?:\.\d+)?\s*(원|주|%)?/gu))
    .map((match) => {
      const raw = match[1] ?? ''
      const value = parseNumber(raw.replace(/^\+/, ''))
      if (!value) return null
      const token: NumericToken = {
        raw: raw.replace(/^\+/, ''),
        value,
      }
      if (match[2]) token.unit = match[2]
      return token
    })
    .filter((token): token is NumericToken => token !== null)
}

function lineAfterDefinitionAlias(
  definition: PortfolioHoldingDefinition,
  line: string,
): string {
  const { compact, indexMap } = compactWithIndexMap(line)
  const aliases = [...definition.aliases].sort((left, right) => {
    return right.replace(/\s+/g, '').length - left.replace(/\s+/g, '').length
  })

  for (const alias of aliases) {
    const compactAlias = alias.toLowerCase().replace(/\s+/g, '')
    if (!compact.startsWith(compactAlias)) continue
    const afterAlias = indexMap[compactAlias.length]
    return line.slice(afterAlias ?? line.length)
  }

  return line
}

function extractHoldingFromOrderedText(
  definition: PortfolioHoldingDefinition,
  text: string,
): PortfolioHolding | null {
  const tokens = parseNumericTokens(text)
    .filter((token) => token.unit !== '%')
    .filter((token) => token.raw.replace(/,/g, '') !== definition.symbol)
  if (tokens.length < 2) return null

  let averageIndex = tokens.findIndex((token) => token.unit === '원')
  if (averageIndex < 0) averageIndex = 0

  const averageCost = tokens[averageIndex]?.value
  const quantity = tokens
    .slice(averageIndex + 1)
    .find((token) => token.unit !== '원')?.value
  if (!averageCost || !quantity) return null

  return {
    label: definition.label,
    symbol: definition.symbol,
    averageCost,
    quantity,
  }
}

function extractHoldingFromDelimitedRow(
  definition: PortfolioHoldingDefinition,
  window: string,
): PortfolioHolding | null {
  const firstLine = window.split(/\r?\n/, 1)[0] ?? ''
  if (!firstLine.includes('|')) return null

  const parts = firstLine.split('|').map((part) => part.trim()).filter(Boolean)
  const compactParts = parts.map((part) => part.toLowerCase().replace(/\s+/g, ''))
  const aliasIndex = compactParts.findIndex((part) =>
    definition.aliases.some((alias) => part.includes(alias.toLowerCase().replace(/\s+/g, ''))),
  )
  if (aliasIndex < 0) return null

  return extractHoldingFromOrderedText(
    definition,
    parts.slice(aliasIndex + 1).join(' '),
  )
}

function extractHoldingFromLabelledText(
  definition: PortfolioHoldingDefinition,
  text: string,
): PortfolioHolding | null {
  const averageCost = parseNumber(
    text.match(/([\d,]+)\s*원?(?:의\s*)?(?:평균\s*)?단가/i)?.[1]
    ?? text.match(/(?:평균\s*단가|평균단가|매입\s*단가|단가|average\s*cost)\s*(?:는|은|가|로|:|\||-)?\s*([\d,]+)\s*원?/i)?.[1],
  )
  const quantity = parseNumber(
    text.match(/(?:보유\s*수량|보유수량|수량|quantity|shares)\s*(?:은|는|:|\||-)?\s*([\d,]+)/i)?.[1]
    ?? text.match(/([\d,]+)\s*주/i)?.[1],
  )
  if (!averageCost || !quantity) return null

  return {
    label: definition.label,
    symbol: definition.symbol,
    averageCost,
    quantity,
  }
}

function extractHoldingFromPlainRow(
  definition: PortfolioHoldingDefinition,
  window: string,
): PortfolioHolding | null {
  const firstLine = window.split(/\r?\n/, 1)[0] ?? window
  const afterAlias = lineAfterDefinitionAlias(definition, firstLine)
  return extractHoldingFromOrderedText(definition, afterAlias)
}

function extractHoldingFromWindow(
  definition: PortfolioHoldingDefinition,
  window: string,
): PortfolioHolding | null {
  return extractHoldingFromDelimitedRow(definition, window)
    ?? extractHoldingFromLabelledText(definition, window.split(/\r?\n/, 1)[0] ?? window)
    ?? extractHoldingFromPlainRow(definition, window)
}

export function extractPortfolioHoldingsFromText(text: string): PortfolioHolding[] {
  const { compact, indexMap } = compactWithIndexMap(text)
  const holdings = new Map<string, PortfolioHolding>()

  for (const definition of PORTFOLIO_HOLDING_DEFINITIONS) {
    for (const alias of definition.aliases) {
      const compactAlias = alias.toLowerCase().replace(/\s+/g, '')
      let searchFrom = 0
      while (searchFrom < compact.length) {
        const compactIndex = compact.indexOf(compactAlias, searchFrom)
        if (compactIndex < 0) break
        const sourceIndex = indexMap[compactIndex] ?? 0
        const window = text.slice(sourceIndex, sourceIndex + 260)
        const holding = extractHoldingFromWindow(definition, window)
        if (holding) {
          holdings.set(definition.symbol, holding)
        }
        searchFrom = compactIndex + compactAlias.length
      }
    }
  }

  return PORTFOLIO_HOLDING_DEFINITIONS
    .map((definition) => holdings.get(definition.symbol))
    .filter((holding): holding is PortfolioHolding => Boolean(holding))
}

function formatKrw(value: number): string {
  return `${new Intl.NumberFormat('ko-KR').format(Math.round(value))}원`
}

export function formatPortfolioHoldingsForMemory(text: string): string | null {
  const holdings = extractPortfolioHoldingsFromText(text)
  if (holdings.length === 0) return null

  return [
    '보유 종목 정보:',
    ...holdings.map((holding) =>
      `- ${holding.label} (${holding.symbol}): 평균단가 ${formatKrw(holding.averageCost)}, 보유수량 ${new Intl.NumberFormat('ko-KR').format(holding.quantity)}주`,
    ),
  ].join('\n')
}
