export interface ExpandDeps {
  args: string
  runCommand: (cmd: string) => Promise<string>
  readFile: (path: string) => Promise<string>
}

export async function expandCommandTemplate(tpl: string, deps: ExpandDeps): Promise<string> {
  const positional = deps.args.split(/\s+/).filter(Boolean)
  let out = tpl
    .replace(/\$ARGUMENTS/g, deps.args)
    .replace(/\$(\d+)/g, (_, n) => positional[Number(n) - 1] ?? '')
  out = await replaceAsync(out, /!`([^`]+)`/g, async (_, cmd) =>
    (await deps.runCommand(cmd)).trim(),
  )
  out = await replaceAsync(out, /@([^\s]+)/g, async (_, path) =>
    (await deps.readFile(path)).trim(),
  )
  return out
}

async function replaceAsync(
  s: string,
  re: RegExp,
  fn: (m: string, g1: string) => Promise<string>,
): Promise<string> {
  const parts: string[] = []
  let last = 0
  let m: RegExpExecArray | null
  re.lastIndex = 0
  while ((m = re.exec(s)) !== null) {
    parts.push(s.slice(last, m.index))
    parts.push(await fn(m[0], m[1]))
    last = m.index + m[0].length
  }
  parts.push(s.slice(last))
  return parts.join('')
}
