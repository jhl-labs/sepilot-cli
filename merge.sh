#!/usr/bin/env bash
# release.sh - Remove sepilot/web/ module and all related references
#
# Usage: bash release.sh [--dry-run]
#
# This script:
#   1. Removes sepilot/web/ directory
#   2. Cleans sepilot/cli/main.py (web server code, --web/--listen options)
#   3. Cleans sepilot/agent/base_agent.py (monitor/shared_memory references)
#   4. Cleans sepilot/agent/approval_handler.py (web_input_queue references)

set -euo pipefail

cp ../sepilot . -rf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No files will be modified."
fi

run() {
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# ------------------------------------------------------------------
# 1. Remove sepilot/web/ directory
# ------------------------------------------------------------------
echo "==> Removing sepilot/web/ directory..."
if [ -d "sepilot/web" ]; then
    run rm -rf sepilot/web
    echo "    Removed sepilot/web/"
else
    echo "    sepilot/web/ not found, skipping."
fi

# ------------------------------------------------------------------
# 2. Clean sepilot/cli/main.py
# ------------------------------------------------------------------
echo "==> Cleaning sepilot/cli/main.py..."

if ! $DRY_RUN; then
python3 << 'PYEOF'
import re

filepath = "sepilot/cli/main.py"
with open(filepath, "r") as f:
    content = f.read()

# --- Remove web/listen params from _run_interactive_mode signature ---
content = re.sub(
    r',\s*\n\s*web:\s*bool\s*=\s*False,\s*\n\s*listen:\s*str\s*=\s*"127\.0\.0\.1"',
    '',
    content
)

# --- Remove the entire "# Wire up web server references" block (if web_server ... pass) ---
content = re.sub(
    r'\n\s*# Wire up web server references if running with --web\n'
    r'(?:.*\n)*?'
    r'\s*pass\s*# Web wiring is best-effort\n',
    '\n',
    content
)

# --- Remove the entire "# Start web server" block through ALL except blocks ---
# Match from "# Start web server" to the LAST "Continuing without web interface"
content = re.sub(
    r'\n\s*# Start web server if --web option is enabled\n'
    r'\s*web_server = None\n'
    r'(?:.*\n)*?'
    r'\s*console\.print\("\[dim\]Continuing without web interface\.\.\.\[/dim\]\\n"\)\n'
    r'(?:\s*except Exception.*\n'
    r'(?:.*\n)*?'
    r'\s*console\.print\("\[dim\]Continuing without web interface\.\.\.\[/dim\]\\n"\)\n)?',
    '\n',
    content
)

# --- Remove the "# Stop web server if running" cleanup block ---
content = re.sub(
    r'\n\s*# Stop web server if running\n'
    r'\s*if web_server:\n'
    r'(?:.*\n)*?'
    r'\s*devnull\.close\(\)\n',
    '\n',
    content
)

# --- Remove --web click option ---
content = re.sub(
    r'@click\.option\(\s*\n\s*"--web",\s*\n\s*is_flag=True,\s*\n\s*help="Enable Web UI \(only works with --interactive mode\)"\s*\n\s*\)\s*\n',
    '',
    content
)

# --- Remove --listen click option ---
content = re.sub(
    r'@click\.option\(\s*\n\s*"--listen",\s*\n\s*default="127\.0\.0\.1",\s*\n\s*help="Host address to bind web server \(default: 127\.0\.0\.1\)\. Use 0\.0\.0\.0 for all interfaces"\s*\n\s*\)\s*\n',
    '',
    content
)

# --- Remove web/listen from main() function signature ---
content = re.sub(r'\s*web:\s*bool,\n', '\n', content)
content = re.sub(r'\s*listen:\s*str,\n', '', content)

# --- Remove web validation block ---
content = re.sub(
    r'\s*# Validate web option requires interactive mode\n'
    r'\s*if web and not interactive:\n'
    r'\s*console\.print\("\[bold red\]Error: --web option requires --interactive mode\[/bold red\]"\)\n'
    r'\s*sys\.exit\(1\)\n',
    '',
    content
)

# --- Remove web=web, listen=listen from _run_interactive_mode call ---
content = re.sub(
    r',\s*\n\s*web=web,\s*\n\s*listen=listen',
    '',
    content
)

with open(filepath, "w") as f:
    f.write(content)

print("    Cleaned sepilot/cli/main.py")
PYEOF
else
    echo "[DRY-RUN] Would clean sepilot/cli/main.py"
fi

# ------------------------------------------------------------------
# 3. Clean sepilot/agent/base_agent.py
# ------------------------------------------------------------------
echo "==> Cleaning sepilot/agent/base_agent.py..."

if ! $DRY_RUN; then
python3 << 'PYEOF'
import re

filepath = "sepilot/agent/base_agent.py"
with open(filepath, "r") as f:
    lines = f.readlines()

out = []
skip_block = False
indent_level = 0

i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.rstrip()

    # Remove self.monitor = None and self.shared_memory = None init lines
    if re.match(r'\s*self\.monitor\s*=\s*None\s*#.*Web', stripped):
        i += 1
        continue
    if re.match(r'\s*self\.shared_memory\s*=\s*None\s*#.*Web', stripped):
        i += 1
        continue

    # Remove "if hasattr(self, 'monitor') and self.monitor:" blocks
    m = re.match(r'^(\s*)if hasattr\(self,\s*[\'"]monitor[\'"]\)\s*and\s*self\.monitor', stripped)
    if m:
        indent_level = len(m.group(1))
        skip_block = True
        i += 1
        # Skip all lines that are more indented (the body of the if block)
        while i < len(lines):
            next_line = lines[i]
            if next_line.strip() == '':
                i += 1
                continue
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent > indent_level:
                i += 1
            else:
                break
        skip_block = False
        continue

    out.append(line)
    i += 1

with open(filepath, "w") as f:
    f.writelines(out)

print("    Cleaned sepilot/agent/base_agent.py")
PYEOF
else
    echo "[DRY-RUN] Would clean sepilot/agent/base_agent.py"
fi

# ------------------------------------------------------------------
# 4. Clean sepilot/agent/approval_handler.py
# ------------------------------------------------------------------
echo "==> Cleaning sepilot/agent/approval_handler.py..."

if ! $DRY_RUN; then
python3 << 'PYEOF'
import re

filepath = "sepilot/agent/approval_handler.py"
with open(filepath, "r") as f:
    lines = f.readlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # 1) Remove web_input_queue param from __init__ signature
    #    e.g. "        web_input_queue: "asyncio.Queue | None" = None,"
    if 'web_input_queue' in stripped and 'asyncio.Queue' in stripped:
        # Ensure previous line ends with comma (fix missing comma)
        if out and out[-1].rstrip().endswith(','):
            pass  # already has comma
        elif out and not out[-1].rstrip().endswith(','):
            out[-1] = out[-1].rstrip() + ',\n'
        i += 1
        continue

    # 2) Remove web_input_queue docstring line
    if 'web_input_queue:' in stripped and 'Optional asyncio Queue' in stripped:
        i += 1
        continue

    # 3) Remove self.web_input_queue assignment
    if re.match(r'\s*self\.web_input_queue\s*=', stripped):
        i += 1
        continue

    # 3b) Fix lines that got merged by prior removal
    #     e.g. "self.auto_approve_session = False  # ...        self.permission_manager = ..."
    #     Split into two properly indented lines
    m = re.match(r'^(\s*)(self\.\w+\s*=\s*\S+.*?)(\s{2,})(self\.\w+\s*=.*)$', line)
    if m:
        indent, first_part, _gap, second_part = m.groups()
        out.append(indent + first_part.rstrip() + '\n')
        line = indent + second_part.lstrip() + '\n'

    # 4) Remove "and self.web_input_queue is None" from conditions
    #    Also fix trailing whitespace left behind
    if 'self.web_input_queue is None' in line:
        line = line.replace(' and self.web_input_queue is None', '')
        line = re.sub(r' +:', ':', line)  # fix "isatty() :" -> "isatty():"

    # 5) Remove web docstring lines in _safe_input + following blank line
    if 'Supports both CLI and web-based input' in stripped:
        i += 1
        # also skip "it will wait for input from both sources concurrently"
        if i < len(lines) and 'it will wait for input from both sources' in lines[i]:
            i += 1
        # also skip blank line that followed the two docstring lines
        if i < len(lines) and lines[i].strip() == '':
            i += 1
        continue

    # 6) Remove "# Check if we have a web input queue" + if block + following blank line
    if '# Check if we have a web input queue' in stripped:
        i += 1  # skip comment
        # skip the "if self.web_input_queue is not None:" line
        if i < len(lines) and 'web_input_queue is not None' in lines[i]:
            i += 1
        # skip the "return self._get_input_with_web_support(prompt)" line
        if i < len(lines) and '_get_input_with_web_support' in lines[i]:
            i += 1
        # skip blank line after the removed block
        if i < len(lines) and lines[i].strip() == '':
            i += 1
        continue

    # 7) Remove entire _get_input_with_web_support method + trailing blank lines
    if re.match(r'\s*def _get_input_with_web_support\(', stripped):
        method_indent = len(line) - len(line.lstrip())
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if next_line.strip() == '':
                i += 1
                continue
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent > method_indent:
                i += 1
            else:
                break
        continue

    out.append(line)
    i += 1

with open(filepath, "w") as f:
    f.writelines(out)

print("    Cleaned sepilot/agent/approval_handler.py")
PYEOF
else
    echo "[DRY-RUN] Would clean sepilot/agent/approval_handler.py"
fi

# ------------------------------------------------------------------
# 5. Verify no remaining references
# ------------------------------------------------------------------
echo ""
echo "==> Checking for remaining web references..."
REMAINING=$(grep -rn "sepilot\.web\|sepilot/web\|from sepilot\.web\|web_input_queue\|WebServer\|ConsoleCapture\|LangGraphMonitor\|ConnectionManager\|SharedMemoryStore" \
    --include="*.py" sepilot/ \
    --exclude-dir="web" \
    --exclude-dir="tools" \
    2>/dev/null || true)

if [ -n "$REMAINING" ]; then
    echo "    [WARNING] Remaining references found:"
    echo "$REMAINING" | head -30
    echo ""
    echo "    Please review and clean up manually if needed."
else
    echo "    No remaining web references found."
fi

echo ""
echo "==> Done! Web module has been removed."
echo "    Run 'git diff' to review changes before committing."
