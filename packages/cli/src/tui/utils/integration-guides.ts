export const ACP_COMMAND_USAGE = [
  'ACP Integration',
  'Usage: /acp [help|status|config|opencode|codex|zed]',
  '',
  'Supported stdio JSON-RPC methods:',
  '- initialize',
  '- session/new, session/prompt, session/cancel',
  '- legacy compatibility: newThread, sendMessage, cancelThread',
  '',
  'CLI entry: sepilot acp [--url <daemon-url>]',
].join('\n')

export const A2A_COMMAND_USAGE = [
  'A2A Integration',
  'Usage: /a2a [help|status|card|send]',
  '',
  'Standard endpoints:',
  '- GET /.well-known/agent-card.json',
  '- POST /api/v1/a2a',
  '',
  'JSON-RPC methods:',
  '- SendMessage, GetTask, ListTasks, CancelTask',
  '- unsupported features return A2A JSON-RPC errors: streaming, push notifications, extended cards',
].join('\n')

export function buildAcpConfigSnippet(daemonUrl?: string): string {
  const args = ['acp']
  if (daemonUrl) args.push('--url', daemonUrl)
  return [
    'ACP editor command:',
    `  sepilot ${args.join(' ')}`,
    '',
    'Generic stdio registration:',
    JSON.stringify({ command: 'sepilot', args, env: daemonUrl ? { SEPILOTD_URL: daemonUrl } : {} }, null, 2),
  ].join('\n')
}

export function buildAcpOpencodeGuide(): string {
  return [
    'opencode ACP adapter path',
    '1. opencode exposes an ACP agent with `opencode acp`.',
    '2. sepilotd exposes its own ACP server with `sepilot acp`.',
    '3. sepilotd can run opencode through the daemon tool `external_acp.run`.',
    '',
    'CLI: sepilot acp-agent run --agent opencode --cwd <project> "<task>"',
  ].join('\n')
}

export function buildAcpCodexGuide(): string {
  return [
    'Codex ACP adapter path',
    'Codex runs through the stdio adapter `@agentclientprotocol/codex-acp`.',
    'Install: npm install -g @agentclientprotocol/codex-acp',
    'CLI: sepilot acp-agent run --agent codex --cwd <project> "<task>"',
  ].join('\n')
}

export function buildAcpZedHint(daemonUrl?: string): string {
  return [
    'Zed / ACP registration hint',
    'Register a stdio agent that runs:',
    `  sepilot acp${daemonUrl ? ` --url ${daemonUrl}` : ''}`,
    '',
    'The daemon must already be reachable. Verify with `/acp status`.',
  ].join('\n')
}

export function buildA2aGuide(daemonUrl?: string): string {
  const base = daemonUrl ?? 'http://<daemon-host>:17600'
  return [
    'A2A endpoint registration',
    `Agent Card: ${base.replace(/\/$/, '')}/.well-known/agent-card.json`,
    `JSON-RPC:   ${base.replace(/\/$/, '')}/api/v1/a2a`,
    '',
    'Protocol: Agent2Agent JSON-RPC binding, A2A-Version: 1.0',
    'Outbound tool: a2a.send',
  ].join('\n')
}
