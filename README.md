# sepilot-cli

`sepilot-cli` provides `sepilot`, a local-first AI agent CLI, and `sepilotd`,
the local daemon that owns sessions, tools, approvals, memory, skills, and MCP
integrations. The CLI communicates with the daemon through HTTP and WebSocket.

## License

This repository is **source-available, not open source** under the
[Sepilot Source-Available License](LICENSE). Individuals and businesses may
use the unmodified software, including commercially. Copying, modifying,
redistributing, or using the source in another product requires the copyright
holder's prior written permission. See [LICENSE](LICENSE) for the complete
terms.

## Install a released binary

macOS and Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.ps1 | iex
```

Each installer downloads the current GitHub Release for the detected platform
and verifies its SHA-256 checksum before installing it. Supported release
assets are Linux (x64, arm64), macOS (x64, arm64), and Windows (x64).

## Build from source

Requirements: Node.js 22, pnpm 10, and Bun (only for a standalone binary).

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm build

# Start from built sources
node packages/daemon/dist/index.js
node packages/cli/dist/index.js --help

# Or build and install a standalone binary for this machine
./install-cli.sh
```

The standalone binary embeds the daemon. `sepilot` starts it as needed;
`sepilot start`, `stop`, `restart`, and `status` manage it explicitly.

## Configuration

Runtime state is stored under `~/.sepilotd/`. Copy
[`config.example.yaml`](config.example.yaml) to `~/.sepilotd/config.yaml` and
provide credentials through environment variables or a local configuration
file. Never commit credentials, tokens, or runtime state.

## Packages

| Package | Purpose |
| --- | --- |
| `@sepilotd/core` | Shared contracts and domain types |
| `@sepilotd/api-client` | Typed daemon HTTP/WebSocket client |
| `@sepilotd/presentation` | CLI artifact and canvas rendering primitives |
| `@sepilotd/daemon` | Local daemon runtime |
| `@sepilotd/cli` | `sepilot` commands and terminal UI |
| `@sepilotd/bundle` | Self-contained `sepilot` binary builder |

## Development

```sh
pnpm typecheck
pnpm build
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull
request. Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md).

## Releases

Pushing a tag such as `v1.0.5` starts the release workflow. It builds binaries
and SHA-256 checksum files for all supported platforms, then creates a GitHub
Release containing those assets.
