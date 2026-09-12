import { detectCliLocale } from '../utils/locale.js'

const COMPLETIONS_COPY = {
  en: {
    unsupportedShell: (shell: string) => `Unsupported shell: ${shell}`,
    supported: 'Supported: bash, zsh, fish',
    usage: 'Usage: sepilot completions --shell bash >> ~/.bashrc',
  },
  ko: {
    unsupportedShell: (shell: string) => `지원되지 않는 셸: ${shell}`,
    supported: '지원: bash, zsh, fish',
    usage: '사용법: sepilot completions --shell bash >> ~/.bashrc',
  },
} as const

export function completionsCommand(options: { shell?: string }) {
  const shell = options.shell ?? detectShell()

  switch (shell) {
    case 'bash':
      console.log(generateBashCompletion())
      break
    case 'zsh':
      console.log(generateZshCompletion())
      break
    case 'fish':
      console.log(generateFishCompletion())
      break
    default: {
      const copy = COMPLETIONS_COPY[detectCliLocale()] ?? COMPLETIONS_COPY.en
      console.log(copy.unsupportedShell(shell))
      console.log(copy.supported)
      console.log(copy.usage)
    }
  }
}

function detectShell(): string {
  const shell = process.env.SHELL ?? ''
  if (shell.includes('zsh')) return 'zsh'
  if (shell.includes('fish')) return 'fish'
  return 'bash'
}

const COMMANDS = ['chat', 'status', 'sessions', 'skills', 'config', 'providers', 'devices', 'memory', 'usage', 'init', 'doctor', 'logs', 'channel', 'version', 'backup', 'restore', 'completions']

function generateBashCompletion(): string {
  return `# sepilot bash completion
_sepilot_completions() {
  local cur="\${COMP_WORDS[COMP_CWORD]}"
  local commands="${COMMANDS.join(' ')}"

  if [ "$COMP_CWORD" -eq 1 ]; then
    COMPREPLY=($(compgen -W "$commands" -- "$cur"))
  elif [ "$COMP_CWORD" -eq 2 ]; then
    case "\${COMP_WORDS[1]}" in
      sessions) COMPREPLY=($(compgen -W "list show delete compact export" -- "$cur")) ;;
      skills) COMPREPLY=($(compgen -W "list installed search create install uninstall update" -- "$cur")) ;;
      memory) COMPREPLY=($(compgen -W "search documents file backlog add status reindex" -- "$cur")) ;;
      usage) COMPREPLY=($(compgen -W "summary daily" -- "$cur")) ;;
      channel) COMPREPLY=($(compgen -W "list pair" -- "$cur")) ;;
    esac
  fi
}
complete -F _sepilot_completions sepilot`
}

function generateZshCompletion(): string {
  return `# sepilot zsh completion
_sepilot() {
  local -a commands
  commands=(${COMMANDS.map(c => `'${c}:${c} command'`).join(' ')})

  _arguments '1:command:->cmds' '*::arg:->args'

  case "$state" in
    cmds) _describe 'command' commands ;;
    args)
      case $words[1] in
        sessions) _values 'subcommand' list show delete compact export ;;
        skills) _values 'subcommand' list installed search create install uninstall update ;;
        memory) _values 'subcommand' search documents file backlog add status reindex ;;
        usage) _values 'subcommand' summary daily ;;
        channel) _values 'subcommand' list pair ;;
      esac ;;
  esac
}
compdef _sepilot sepilot`
}

function generateFishCompletion(): string {
  const lines = COMMANDS.map(c => `complete -c sepilot -n '__fish_use_subcommand' -a '${c}' -d '${c}'`)
  lines.push("complete -c sepilot -n '__fish_seen_subcommand_from sessions' -a 'list show delete compact export'")
  lines.push("complete -c sepilot -n '__fish_seen_subcommand_from skills' -a 'list installed search create install uninstall update'")
  lines.push("complete -c sepilot -n '__fish_seen_subcommand_from memory' -a 'search documents file backlog add status reindex'")
  return `# sepilot fish completion\n${lines.join('\n')}`
}
