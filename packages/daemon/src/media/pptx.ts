import { posix } from 'node:path'
import { inflateRawSync } from 'node:zlib'

const ZIP_EOCD_SIGNATURE = 0x06054b50
const ZIP_CENTRAL_SIGNATURE = 0x02014b50
const ZIP_LOCAL_SIGNATURE = 0x04034b50

const MAX_PPTX_ENTRIES = 4_096
const MAX_PPTX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
const MAX_PPTX_XML_ENTRY_BYTES = 4 * 1024 * 1024
const MAX_PPTX_XML_TOTAL_BYTES = 32 * 1024 * 1024
const MAX_PPTX_SLIDES = 500
const MAX_PPTX_OUTPUT_CHARS = 500_000

interface ZipEntry {
  name: string
  flags: number
  compressionMethod: number
  compressedSize: number
  uncompressedSize: number
  localHeaderOffset: number
}

interface PptxSlide {
  index: number
  title?: string
  body: string[]
  tables: string[][][]
  notes?: string
}

export class PptxExtractionError extends Error {
  constructor(
    message: string,
    readonly code: 'INVALID_PPTX' | 'PPTX_LIMIT_EXCEEDED',
  ) {
    super(message)
    this.name = 'PptxExtractionError'
  }
}

function invalidPptx(message: string): never {
  throw new PptxExtractionError(`Invalid PPTX archive: ${message}`, 'INVALID_PPTX')
}

function pptxLimit(message: string): never {
  throw new PptxExtractionError(`PPTX safety limit exceeded: ${message}`, 'PPTX_LIMIT_EXCEEDED')
}

function findEndOfCentralDirectory(archive: Buffer): number {
  // EOCD is 22 bytes plus at most a 65,535-byte ZIP comment.
  const floor = Math.max(0, archive.length - 22 - 0xffff)
  for (let offset = archive.length - 22; offset >= floor; offset -= 1) {
    if (archive.readUInt32LE(offset) !== ZIP_EOCD_SIGNATURE) continue
    const commentLength = archive.readUInt16LE(offset + 20)
    if (offset + 22 + commentLength === archive.length) return offset
  }
  return invalidPptx('end-of-central-directory record is missing')
}

function decodeZipEntryName(value: Buffer, utf8: boolean): string {
  // OOXML part names are ASCII. latin1 is intentionally used for legacy ZIP
  // names so decoding can never replace bytes into a misleading slash.
  return value.toString(utf8 ? 'utf8' : 'latin1')
}

function readZipDirectory(archive: Buffer): Map<string, ZipEntry> {
  if (archive.length < 22) invalidPptx('archive is truncated')
  const eocdOffset = findEndOfCentralDirectory(archive)
  const diskNumber = archive.readUInt16LE(eocdOffset + 4)
  const centralDisk = archive.readUInt16LE(eocdOffset + 6)
  const entriesOnDisk = archive.readUInt16LE(eocdOffset + 8)
  const entryCount = archive.readUInt16LE(eocdOffset + 10)
  const centralSize = archive.readUInt32LE(eocdOffset + 12)
  const centralOffset = archive.readUInt32LE(eocdOffset + 16)

  if (diskNumber !== 0 || centralDisk !== 0 || entriesOnDisk !== entryCount) {
    invalidPptx('multi-disk ZIP archives are not supported')
  }
  if (entryCount === 0xffff || centralSize === 0xffffffff || centralOffset === 0xffffffff) {
    invalidPptx('ZIP64 archives are not supported')
  }
  if (entryCount > MAX_PPTX_ENTRIES) {
    pptxLimit(`${entryCount} entries exceeds the ${MAX_PPTX_ENTRIES} entry limit`)
  }
  if (centralOffset + centralSize > eocdOffset || centralOffset > archive.length) {
    invalidPptx('central directory points outside the archive')
  }

  const entries = new Map<string, ZipEntry>()
  let cursor = centralOffset
  let declaredUncompressedBytes = 0
  for (let index = 0; index < entryCount; index += 1) {
    if (cursor + 46 > archive.length || archive.readUInt32LE(cursor) !== ZIP_CENTRAL_SIGNATURE) {
      invalidPptx('central directory entry is truncated or malformed')
    }
    const flags = archive.readUInt16LE(cursor + 8)
    const compressionMethod = archive.readUInt16LE(cursor + 10)
    const compressedSize = archive.readUInt32LE(cursor + 20)
    const uncompressedSize = archive.readUInt32LE(cursor + 24)
    const nameLength = archive.readUInt16LE(cursor + 28)
    const extraLength = archive.readUInt16LE(cursor + 30)
    const commentLength = archive.readUInt16LE(cursor + 32)
    const localHeaderOffset = archive.readUInt32LE(cursor + 42)
    const next = cursor + 46 + nameLength + extraLength + commentLength
    if (next > archive.length) invalidPptx('central directory name or metadata is truncated')

    const name = decodeZipEntryName(
      archive.subarray(cursor + 46, cursor + 46 + nameLength),
      (flags & 0x800) !== 0,
    )
    if (!name || name.includes('\0') || name.includes('\\')) {
      invalidPptx('entry contains an invalid part name')
    }
    if (entries.has(name)) invalidPptx(`duplicate ZIP entry: ${name}`)

    declaredUncompressedBytes += uncompressedSize
    if (declaredUncompressedBytes > MAX_PPTX_UNCOMPRESSED_BYTES) {
      pptxLimit(
        `declared uncompressed content exceeds ${MAX_PPTX_UNCOMPRESSED_BYTES} bytes`,
      )
    }
    entries.set(name, {
      name,
      flags,
      compressionMethod,
      compressedSize,
      uncompressedSize,
      localHeaderOffset,
    })
    cursor = next
  }
  if (cursor !== centralOffset + centralSize) {
    invalidPptx('central directory size does not match its entries')
  }
  return entries
}

function readZipEntry(archive: Buffer, entry: ZipEntry): Buffer {
  if ((entry.flags & 0x1) !== 0) invalidPptx(`encrypted part is not supported: ${entry.name}`)
  if (entry.uncompressedSize > MAX_PPTX_XML_ENTRY_BYTES) {
    pptxLimit(`${entry.name} exceeds the ${MAX_PPTX_XML_ENTRY_BYTES} byte XML part limit`)
  }
  const offset = entry.localHeaderOffset
  if (offset + 30 > archive.length || archive.readUInt32LE(offset) !== ZIP_LOCAL_SIGNATURE) {
    invalidPptx(`local header is missing for ${entry.name}`)
  }
  const localMethod = archive.readUInt16LE(offset + 8)
  const nameLength = archive.readUInt16LE(offset + 26)
  const extraLength = archive.readUInt16LE(offset + 28)
  const dataOffset = offset + 30 + nameLength + extraLength
  const dataEnd = dataOffset + entry.compressedSize
  if (dataEnd > archive.length) invalidPptx(`compressed data is truncated for ${entry.name}`)
  if (localMethod !== entry.compressionMethod) {
    invalidPptx(`compression metadata does not match for ${entry.name}`)
  }

  const compressed = archive.subarray(dataOffset, dataEnd)
  let output: Buffer
  if (entry.compressionMethod === 0) {
    output = Buffer.from(compressed)
  } else if (entry.compressionMethod === 8) {
    try {
      output = inflateRawSync(compressed, {
        // Do not trust the deflate stream even after checking central-directory
        // sizes. One extra byte lets us distinguish an honest size mismatch.
        maxOutputLength: entry.uncompressedSize + 1,
      })
    } catch (error) {
      invalidPptx(
        `cannot decompress ${entry.name}: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
  } else {
    invalidPptx(`unsupported compression method ${entry.compressionMethod} for ${entry.name}`)
  }
  if (output.length !== entry.uncompressedSize) {
    invalidPptx(`uncompressed size does not match for ${entry.name}`)
  }
  return output
}

function decodeXmlEntities(value: string): string {
  return value.replace(
    /&(?:#(\d+)|#x([0-9a-f]+)|([a-z]+));/giu,
    (entity, decimal: string | undefined, hexadecimal: string | undefined, named: string | undefined) => {
      if (decimal) {
        const codePoint = Number.parseInt(decimal, 10)
        return Number.isSafeInteger(codePoint) && codePoint <= 0x10ffff
          ? String.fromCodePoint(codePoint)
          : entity
      }
      if (hexadecimal) {
        const codePoint = Number.parseInt(hexadecimal, 16)
        return Number.isSafeInteger(codePoint) && codePoint <= 0x10ffff
          ? String.fromCodePoint(codePoint)
          : entity
      }
      switch (named?.toLowerCase()) {
        case 'amp': return '&'
        case 'lt': return '<'
        case 'gt': return '>'
        case 'quot': return '"'
        case 'apos': return "'"
        default: return entity
      }
    },
  )
}

function parseAttributes(raw: string): Record<string, string> {
  const attributes: Record<string, string> = {}
  const pattern = /([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/gu
  for (const match of raw.matchAll(pattern)) {
    attributes[match[1]] = decodeXmlEntities(match[2] ?? match[3] ?? '')
  }
  return attributes
}

function paragraphText(fragment: string): string[] {
  const paragraphs: string[] = []
  for (const paragraph of fragment.matchAll(/<a:p(?:\s[^>]*)?>([\s\S]*?)<\/a:p>/gu)) {
    const tokens: string[] = []
    const tokenPattern = /<a:t(?:\s[^>]*)?>([\s\S]*?)<\/a:t>|<a:(br|tab)\b[^>]*\/?\s*>/gu
    for (const token of paragraph[1].matchAll(tokenPattern)) {
      if (token[1] !== undefined) tokens.push(decodeXmlEntities(token[1]))
      else if (token[2] === 'br') tokens.push('\n')
      else tokens.push('\t')
    }
    const text = tokens.join('').replace(/\r/gu, '').trim()
    if (text) paragraphs.push(text)
  }
  return paragraphs
}

function placeholderType(shapeXml: string): string | undefined {
  const match = /<p:ph\b([^>]*)\/?\s*>/u.exec(shapeXml)
  return match ? parseAttributes(match[1]).type : undefined
}

function extractSlideXml(xml: string, index: number): PptxSlide {
  let title: string | undefined
  const body: string[] = []
  const ignoredPlaceholderTypes = new Set(['dt', 'ftr', 'sldNum'])
  for (const shape of xml.matchAll(/<p:sp(?:\s[^>]*)?>([\s\S]*?)<\/p:sp>/gu)) {
    const shapeXml = shape[1]
    const paragraphs = paragraphText(shapeXml)
    if (paragraphs.length === 0) continue
    const type = placeholderType(shapeXml)
    if (type && ignoredPlaceholderTypes.has(type)) continue
    if (type === 'title' || type === 'ctrTitle') {
      title = paragraphs.join(' ').trim()
    } else {
      body.push(paragraphs.join('\n'))
    }
  }

  const tables: string[][][] = []
  for (const table of xml.matchAll(/<a:tbl(?:\s[^>]*)?>([\s\S]*?)<\/a:tbl>/gu)) {
    const rows: string[][] = []
    for (const row of table[1].matchAll(/<a:tr(?:\s[^>]*)?>([\s\S]*?)<\/a:tr>/gu)) {
      const cells: string[] = []
      for (const cell of row[1].matchAll(/<a:tc(?:\s[^>]*)?>([\s\S]*?)<\/a:tc>/gu)) {
        cells.push(paragraphText(cell[1]).join('\n'))
      }
      if (cells.length > 0) rows.push(cells)
    }
    if (rows.length > 0) tables.push(rows)
  }
  return { index, title, body, tables }
}

function extractNotesXml(xml: string): string | undefined {
  const notes: string[] = []
  const ignoredPlaceholderTypes = new Set(['sldImg', 'dt', 'ftr', 'sldNum', 'hdr'])
  for (const shape of xml.matchAll(/<p:sp(?:\s[^>]*)?>([\s\S]*?)<\/p:sp>/gu)) {
    const type = placeholderType(shape[1])
    if (type && ignoredPlaceholderTypes.has(type)) continue
    const paragraphs = paragraphText(shape[1])
    if (paragraphs.length > 0) notes.push(paragraphs.join('\n'))
  }
  const value = notes.join('\n').trim()
  return value || undefined
}

function relationshipMap(xml: string): Map<string, { target: string; type: string }> {
  const relationships = new Map<string, { target: string; type: string }>()
  for (const relationship of xml.matchAll(/<Relationship\b([^>]*)\/?\s*>/gu)) {
    const attrs = parseAttributes(relationship[1])
    if (!attrs.Id || !attrs.Target || attrs.TargetMode === 'External') continue
    relationships.set(attrs.Id, { target: attrs.Target, type: attrs.Type ?? '' })
  }
  return relationships
}

function resolvePartTarget(sourcePart: string, target: string): string | null {
  if (!target || target.startsWith('/') || target.includes('\\') || target.includes('\0')) return null
  const resolved = posix.normalize(posix.join(posix.dirname(sourcePart), target))
  if (resolved === '..' || resolved.startsWith('../')) return null
  return resolved
}

function presentationSlideParts(
  readXml: (name: string, required?: boolean) => string | null,
  entries: Map<string, ZipEntry>,
): string[] {
  const presentation = readXml('ppt/presentation.xml', true)!
  const relationships = relationshipMap(
    readXml('ppt/_rels/presentation.xml.rels', true)!,
  )
  const ordered: string[] = []
  for (const slideId of presentation.matchAll(/<p:sldId\b([^>]*)\/?\s*>/gu)) {
    const attrs = parseAttributes(slideId[1])
    const relationship = relationships.get(attrs['r:id'] ?? '')
    if (!relationship?.type.endsWith('/slide')) continue
    const part = resolvePartTarget('ppt/presentation.xml', relationship.target)
    if (part && entries.has(part)) ordered.push(part)
  }
  if (ordered.length > 0) return [...new Set(ordered)]

  return [...entries.keys()]
    .filter((name) => /^ppt\/slides\/slide\d+\.xml$/u.test(name))
    .sort((left, right) => {
      const leftNumber = Number(/slide(\d+)\.xml$/u.exec(left)?.[1] ?? 0)
      const rightNumber = Number(/slide(\d+)\.xml$/u.exec(right)?.[1] ?? 0)
      return leftNumber - rightNumber
    })
}

function notesPartForSlide(
  slidePart: string,
  readXml: (name: string, required?: boolean) => string | null,
): string | null {
  const relationshipsPart = posix.join(
    posix.dirname(slidePart),
    '_rels',
    `${posix.basename(slidePart)}.rels`,
  )
  const relationshipsXml = readXml(relationshipsPart)
  if (!relationshipsXml) return null
  for (const relationship of relationshipMap(relationshipsXml).values()) {
    if (!relationship.type.endsWith('/notesSlide')) continue
    return resolvePartTarget(slidePart, relationship.target)
  }
  return null
}

function formatTable(rows: string[][]): string[] {
  return rows.map((row) => `| ${row.map((cell) => cell
    .replace(/\\/gu, '\\\\')
    .replace(/\|/gu, '\\|')).join(' | ')} |`)
}

function formatSlides(filename: string, slides: PptxSlide[]): string {
  const lines = [`[PowerPoint: ${filename}]`, `Slides: ${slides.length}`]
  let truncated = false
  for (const slide of slides) {
    const block: string[] = [
      '',
      `## Slide ${slide.index}${slide.title ? ` — ${slide.title}` : ''}`,
    ]
    if (slide.body.length > 0) block.push('', ...slide.body)
    slide.tables.forEach((table, tableIndex) => {
      block.push('', `### Table ${tableIndex + 1}`, ...formatTable(table))
    })
    if (slide.notes) block.push('', '### Speaker notes', slide.notes)
    const candidate = [...lines, ...block].join('\n')
    if (candidate.length > MAX_PPTX_OUTPUT_CHARS) {
      truncated = true
      break
    }
    lines.push(...block)
  }
  if (truncated) {
    lines.push('', `[PPTX extraction truncated at ${MAX_PPTX_OUTPUT_CHARS} characters.]`)
  }
  return lines.join('\n')
}

export function extractPptxText(archive: Buffer, filename = 'presentation.pptx'): string {
  const entries = readZipDirectory(archive)
  let extractedXmlBytes = 0
  const xmlCache = new Map<string, string>()
  const readXml = (name: string, required = false): string | null => {
    const cached = xmlCache.get(name)
    if (cached !== undefined) return cached
    const entry = entries.get(name)
    if (!entry) {
      if (required) invalidPptx(`required OOXML part is missing: ${name}`)
      return null
    }
    extractedXmlBytes += entry.uncompressedSize
    if (extractedXmlBytes > MAX_PPTX_XML_TOTAL_BYTES) {
      pptxLimit(`extracted XML exceeds ${MAX_PPTX_XML_TOTAL_BYTES} bytes`)
    }
    const value = readZipEntry(archive, entry).toString('utf8')
    xmlCache.set(name, value)
    return value
  }

  const slideParts = presentationSlideParts(readXml, entries)
  if (slideParts.length === 0) invalidPptx('presentation contains no slides')
  if (slideParts.length > MAX_PPTX_SLIDES) {
    pptxLimit(`${slideParts.length} slides exceeds the ${MAX_PPTX_SLIDES} slide limit`)
  }

  const slides = slideParts.map((slidePart, offset) => {
    const slide = extractSlideXml(readXml(slidePart, true)!, offset + 1)
    const notesPart = notesPartForSlide(slidePart, readXml)
    if (notesPart) {
      const notesXml = readXml(notesPart)
      if (notesXml) slide.notes = extractNotesXml(notesXml)
    }
    return slide
  })
  return formatSlides(filename, slides)
}
