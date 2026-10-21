export type OutputFormat = 'text' | 'json' | 'stream-json'

let format: OutputFormat = 'text'
export function writeOutputText(value: string): void {
  process.stdout.write(value.endsWith('\n') ? value : `${value}\n`)
}

export function setOutputFormat(f: OutputFormat): void {
  format = f
}

export function getOutputFormat(): OutputFormat {
  return format
}

export function output<T>(data: T, textFormatter?: (data: T) => string): void {
  if (format === 'json') {
    writeJson(data)
  } else if (format === 'stream-json') {
    writeOutputText(JSON.stringify(data))
  } else if (textFormatter) {
    console.log(textFormatter(data))
  } else {
    console.log(data)
  }
}

export function outputError<T>(data: T, textFormatter: (data: T) => string): void {
  if (format === 'json') {
    writeJson(data)
  } else if (format === 'stream-json') {
    writeOutputText(JSON.stringify(data))
  } else {
    console.error(textFormatter(data))
  }
}

function writeJson(data: unknown): void {
  writeOutputText(JSON.stringify(data, null, 2))
}
