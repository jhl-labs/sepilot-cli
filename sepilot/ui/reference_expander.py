"""Reference expansion for @ syntax (Claude Code style).

This module follows the Single Responsibility Principle (SRP) by handling
only @ reference parsing and content expansion.

Supports:
    @file.py          - Single file content
    @folder/          - Directory listing and contents
    @**/*.py          - Glob pattern matching
    @url:https://     - Web page content
    @git:diff         - Git diff output
    @git:HEAD~1       - Git commit info
    @agent:explore    - Agent invocation (new)
    @agent:coder      - Agent invocation (new)
"""

import glob as glob_module
import ipaddress
import os
import re
import socket
import subprocess
from typing import Any
from urllib.parse import urlparse

from rich.console import Console


class ReferenceExpander:
    """Expands @ references by including content in prompts.

    This class handles parsing and expanding various reference types
    in user input. It's designed to be stateless and reusable.
    """

    # Default limits for content inclusion
    DEFAULT_MAX_CHARS_PER_ITEM = 200000
    DEFAULT_MAX_TOTAL_CHARS = 400000

    def __init__(
        self,
        console: Console | None = None,
        max_chars_per_item: int | None = None,
        max_total_chars: int | None = None,
    ):
        """Initialize reference expander.

        Args:
            console: Rich console for output messages
            max_chars_per_item: Max chars per single reference
            max_total_chars: Max total chars for all references
        """
        self.console = console or Console()
        self.max_chars_per_item = max_chars_per_item or int(
            os.getenv("SEPILOT_FILE_REF_MAX_CHARS", str(self.DEFAULT_MAX_CHARS_PER_ITEM))
        )
        self.max_total_chars = max_total_chars or int(
            os.getenv("SEPILOT_FILE_REF_MAX_TOTAL", str(self.DEFAULT_MAX_TOTAL_CHARS))
        )

    def parse_references(self, text: str) -> list[tuple[str, str]]:
        """Parse all @ references from user input.

        Args:
            text: User input text

        Returns:
            List of (type, value) tuples:
            - ("file", "path/to/file.py")
            - ("folder", "path/to/folder")
            - ("glob", "**/*.py")
            - ("url", "https://example.com")
            - ("git", "diff" or "HEAD~1" etc.)
            - ("agent", "explore" or "coder" etc.)
        """
        references: list[tuple[str, str]] = []

        # @agent:explore, @agent:coder, etc.
        agent_pattern = r'@agent:([a-zA-Z_][a-zA-Z0-9_]*)'
        for match in re.finditer(agent_pattern, text):
            references.append(("agent", match.group(1)))

        # @url:https://... or @url:http://...
        url_pattern = r'@url:(https?://[^\s]+)'
        for match in re.finditer(url_pattern, text):
            references.append(("url", match.group(1)))

        # @git:diff, @git:HEAD~1, @git:abc123, etc.
        git_pattern = r'@git:([^\s]+)'
        for match in re.finditer(git_pattern, text):
            references.append(("git", match.group(1)))

        # @folder/ (trailing slash indicates directory)
        folder_pattern = r'@([\w\-_/\.]+)/'
        for match in re.finditer(folder_pattern, text):
            path = match.group(1)
            if not path.startswith(('url:', 'git:')):
                references.append(("folder", path))

        # @**/*.py or @src/**/*.ts (glob patterns)
        glob_pattern = r'@(\*\*?[^\s]+|\S+\*\S*)'
        for match in re.finditer(glob_pattern, text):
            pattern = match.group(1)
            if '*' in pattern and not pattern.startswith(('url:', 'git:')):
                references.append(("glob", pattern))

        # @file.py (regular files)
        file_pattern = r'@([\w\-_/\.]+)'
        for match in re.finditer(file_pattern, text):
            path = match.group(1)
            if (path.startswith(('url:', 'git:', 'agent:')) or
                '*' in path or
                any(ref[1] == path for ref in references)):
                continue
            # Skip if this is part of an @agent:name reference
            match_start = match.start()
            if match_start > 0 and text[match_start - 1:match_start] != '@':
                # Check if this is part of @agent:name where 'agent' was extracted
                if path == 'agent':
                    continue
            match_end = match.end()
            if match_end < len(text) and text[match_end] == '/':
                continue
            if match_end < len(text) and text[match_end] == ':':
                # This is a prefix like @git: or @url: or @agent:
                continue
            references.append(("file", path))

        return references

    def parse_file_references(self, text: str) -> list[str]:
        """Parse @file references only (legacy compatibility).

        Args:
            text: User input text

        Returns:
            List of file paths
        """
        refs = self.parse_references(text)
        return [path for ref_type, path in refs if ref_type == "file"]

    @staticmethod
    def _is_private_or_local_host(host: str | None) -> bool:
        """Check whether a host points to local/private network targets."""
        if not host:
            return True

        normalized = host.strip().lower().strip("[]")
        if normalized in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:  # nosec B104
            return True
        if normalized.endswith((".local", ".localdomain", ".internal")):
            return True

        try:
            ip = ipaddress.ip_address(normalized)
            return any([
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            ])
        except ValueError:
            return False

    def _is_safe_external_url(self, url: str) -> tuple[bool, str]:
        """Validate URL scheme and host for safer fetching."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False, "Only http/https URLs are allowed"
        if not parsed.netloc:
            return False, "URL host is missing"

        allow_private_urls = os.getenv("SEPILOT_ALLOW_PRIVATE_URLS", "").strip() == "1"
        if not allow_private_urls:
            hostname = parsed.hostname
            if self._is_private_or_local_host(hostname):
                return False, "Private/local network URLs are blocked by default"
            # DNS rebinding/alias protection: block hostnames that resolve to
            # loopback/private/link-local ranges even if hostname looks public.
            if hostname:
                try:
                    addr_info = socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
                    for _, _, _, _, sockaddr in addr_info:
                        ip = sockaddr[0]
                        if self._is_private_or_local_host(ip):
                            return False, f"Hostname resolves to private/local address: {ip}"
                except socket.gaierror:
                    # Keep behavior best-effort: unreachable DNS should not crash expansion.
                    pass

        return True, ""

    def expand_references(self, user_input: str) -> str:
        """Expand all @ references by including contents in the prompt.

        Args:
            user_input: Original user input with @ references

        Returns:
            Expanded prompt with all referenced contents
        """
        references = self.parse_references(user_input)

        if not references:
            return user_input

        # Remove @ references from the original input
        cleaned_input = re.sub(r'@(url:|git:)?[\w\-_/\.\*:]+/?', '', user_input).strip()

        # Build the expanded prompt
        expanded_parts: list[str] = []
        total_chars = 0

        for ref_type, ref_value in references:
            if total_chars >= self.max_total_chars:
                self.console.print(
                    f"[yellow]⚠️  Context budget reached ({self.max_total_chars} chars). "
                    "Skipping remaining references.[/yellow]"
                )
                break

            remaining_budget = self.max_total_chars - total_chars

            try:
                content = self._expand_single_reference(
                    ref_type, ref_value, min(self.max_chars_per_item, remaining_budget)
                )
                if content:
                    total_chars += len(content)
                    expanded_parts.append(content)

            except Exception as e:
                self.console.print(f"[red]❌ Error expanding @{ref_type}:{ref_value}: {e}[/red]")

        # Combine: referenced contents + user instruction
        if expanded_parts:
            context = "\n\n".join(expanded_parts)
            return f"{context}\n\n{cleaned_input}"

        return user_input

    def _expand_single_reference(
        self, ref_type: str, ref_value: str, max_chars: int
    ) -> str | None:
        """Expand a single reference based on its type.

        Args:
            ref_type: Type of reference (file, folder, glob, url, git, agent)
            ref_value: The reference value
            max_chars: Maximum characters to include

        Returns:
            Expanded content or None
        """
        expanders = {
            "file": self._expand_file_content,
            "folder": self._expand_folder_content,
            "glob": self._expand_glob_content,
            "url": self._expand_url_content,
            "git": self._expand_git_content,
            "agent": self._expand_agent_reference,
        }

        expander = expanders.get(ref_type)
        if expander:
            return expander(ref_value, max_chars)
        return None

    def _expand_agent_reference(self, agent_name: str, max_chars: int) -> str | None:
        """Expand an agent reference (marks for later invocation).

        Agent references are not expanded inline but marked for the agent system
        to handle specially.

        Args:
            agent_name: Name of the agent (explore, coder, etc.)
            max_chars: Maximum characters (not used for agents)

        Returns:
            Agent invocation marker
        """
        # Available agents
        known_agents = {
            "explore": "Codebase exploration and analysis",
            "coder": "Code generation and implementation",
            "reviewer": "Code review and suggestions",
            "refactor": "Code refactoring",
            "docs": "Documentation generation",
            "test": "Test generation",
            "debug": "Debugging assistance",
        }

        if agent_name.lower() not in known_agents:
            self.console.print(
                f"[yellow]Unknown agent '{agent_name}'. "
                f"Available: {', '.join(known_agents.keys())}[/yellow]"
            )
            return None

        description = known_agents.get(agent_name.lower(), "")
        self.console.print(
            f"[dim cyan]Invoking @agent:{agent_name} ({description})[/dim cyan]"
        )

        # Return a special marker that the agent system will recognize
        return f"__AGENT_INVOKE__:{agent_name}"

    def _expand_file_content(self, file_path: str, max_chars: int) -> str | None:
        """Expand a single file reference."""
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            self.console.print(f"[yellow]⚠️  File not found: {file_path}[/yellow]")
            return None

        if os.path.isdir(abs_path):
            self.console.print(f"[yellow]⚠️  Use @{file_path}/ for directories[/yellow]")
            return None

        content = self._read_file_content(abs_path, file_path)
        if content is None:
            return None

        if len(content) > max_chars:
            self.console.print(f"[yellow]⚠️  Truncating {file_path} to {max_chars} chars[/yellow]")
            content = content[:max_chars] + "\n\n[Content truncated]"

        self.console.print(f"[dim cyan]📎 File: {file_path} ({len(content)} chars)[/dim cyan]")
        return f"File: {file_path}\n```\n{content}\n```"

    def _expand_folder_content(self, folder_path: str, max_chars: int) -> str | None:
        """Expand a folder reference."""
        abs_path = os.path.abspath(folder_path)

        if not os.path.exists(abs_path):
            self.console.print(f"[yellow]⚠️  Folder not found: {folder_path}[/yellow]")
            return None

        if not os.path.isdir(abs_path):
            self.console.print(f"[yellow]⚠️  Not a directory: {folder_path}[/yellow]")
            return None

        # Get file list
        files = self._get_directory_files(abs_path, folder_path)
        if files is None:
            return None

        # Build content
        content_parts = [f"Directory: {folder_path}/", f"Files ({len(files)}):"]
        for f in files:
            content_parts.append(f"  - {f}")

        # Include small text files if budget allows
        file_contents = self._get_small_file_contents(abs_path, files, max_chars, len('\n'.join(content_parts)))

        content = '\n'.join(content_parts)
        if file_contents:
            content += "\n\nFile Contents:" + ''.join(file_contents)

        self.console.print(f"[dim cyan]📁 Folder: {folder_path}/ ({len(files)} files, {len(content)} chars)[/dim cyan]")
        return content

    def _get_directory_files(self, abs_path: str, display_path: str) -> list[str] | None:
        """Get list of files in a directory."""
        files: list[str] = []
        ignored_dirs = {'node_modules', '__pycache__', 'venv', 'env', '.git', 'dist', 'build'}

        try:
            for root, dirs, filenames in os.walk(abs_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ignored_dirs]

                for filename in filenames:
                    if filename.startswith('.'):
                        continue
                    rel_path = os.path.relpath(os.path.join(root, filename), abs_path)
                    files.append(rel_path)

                if root.count(os.sep) - abs_path.count(os.sep) >= 3:
                    dirs.clear()

            files.sort()
            if len(files) > 100:
                files = files[:100]
                self.console.print(f"[yellow]⚠️  Showing first 100 files in {display_path}/[/yellow]")

            return files

        except Exception as e:
            self.console.print(f"[red]❌ Error listing {display_path}: {e}[/red]")
            return None

    def _get_small_file_contents(
        self, abs_path: str, files: list[str], max_chars: int, chars_used: int
    ) -> list[str]:
        """Get contents of small text files within budget."""
        file_contents: list[str] = []
        max_files_to_include = 10
        included_count = 0
        text_extensions = ('.py', '.js', '.ts', '.json', '.yaml', '.yml', '.md', '.txt', '.toml', '.cfg', '.ini')

        for f in files:
            if included_count >= max_files_to_include or chars_used >= max_chars:
                break

            full_path = os.path.join(abs_path, f)
            try:
                if os.path.getsize(full_path) > 10000:
                    continue
            except OSError:
                continue

            if not f.endswith(text_extensions):
                continue

            file_content = self._read_file_content(full_path, f)
            if file_content and len(file_content) < 5000:
                file_entry = f"\n--- {f} ---\n{file_content}"
                if chars_used + len(file_entry) <= max_chars:
                    file_contents.append(file_entry)
                    chars_used += len(file_entry)
                    included_count += 1

        return file_contents

    def _expand_glob_content(self, pattern: str, max_chars: int) -> str | None:
        """Expand a glob pattern reference."""
        try:
            matches = glob_module.glob(pattern, recursive=True)
            matches = [f for f in matches if os.path.isfile(f)]
            matches.sort()

            if not matches:
                self.console.print(f"[yellow]⚠️  No files match pattern: {pattern}[/yellow]")
                return None

            if len(matches) > 50:
                self.console.print(
                    f"[yellow]⚠️  Pattern {pattern} matched {len(matches)} files, showing first 50[/yellow]"
                )
                matches = matches[:50]

            content_parts = [f"Glob Pattern: {pattern}", f"Matched Files ({len(matches)}):"]
            chars_used = 0

            for filepath in matches:
                file_content = self._read_file_content(filepath, filepath)
                if file_content:
                    entry = f"\n--- {filepath} ---\n```\n{file_content}\n```"
                    if chars_used + len(entry) > max_chars:
                        content_parts.append(f"\n[Remaining {len(matches) - len(content_parts) + 2} files truncated]")
                        break
                    content_parts.append(entry)
                    chars_used += len(entry)

            content = '\n'.join(content_parts)
            self.console.print(f"[dim cyan]🔍 Glob: {pattern} ({len(matches)} files, {len(content)} chars)[/dim cyan]")
            return content

        except Exception as e:
            self.console.print(f"[red]❌ Error with glob pattern {pattern}: {e}[/red]")
            return None

    def _expand_url_content(self, url: str, max_chars: int) -> str | None:
        """Expand a URL reference by fetching web page content."""
        try:
            import urllib.error
            import urllib.request
            from html.parser import HTMLParser

            class HTMLTextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts: list[str] = []
                    self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
                    self.current_skip = False

                def handle_starttag(self, tag: str, attrs: Any) -> None:
                    if tag.lower() in self.skip_tags:
                        self.current_skip = True

                def handle_endtag(self, tag: str) -> None:
                    if tag.lower() in self.skip_tags:
                        self.current_skip = False

                def handle_data(self, data: str) -> None:
                    if not self.current_skip:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

                def get_text(self) -> str:
                    return '\n'.join(self.text_parts)

            is_safe, reason = self._is_safe_external_url(url)
            if not is_safe:
                self.console.print(f"[red]❌ URL blocked: {reason} ({url})[/red]")
                return None

            self.console.print(f"[dim]🌐 Fetching URL: {url}...[/dim]")

            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'SEPilot/1.0 (CLI Agent)'}
            )

            with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
                final_url = url
                if hasattr(response, "geturl"):
                    redirect_target = response.geturl()
                    if isinstance(redirect_target, str) and redirect_target:
                        final_url = redirect_target
                is_safe_final, final_reason = self._is_safe_external_url(final_url)
                if not is_safe_final:
                    self.console.print(f"[red]❌ Redirect blocked: {final_reason} ({final_url})[/red]")
                    return None

                content_type = response.headers.get('Content-Type', '')
                max_bytes = min(max_chars * 4, 2_000_000)
                raw_content = response.read(max_bytes + 1)
                was_truncated = len(raw_content) > max_bytes
                if was_truncated:
                    raw_content = raw_content[:max_bytes]

                encoding = 'utf-8'
                if 'charset=' in content_type:
                    encoding = content_type.split('charset=')[-1].split(';')[0].strip()

                try:
                    html_content = raw_content.decode(encoding)
                except (UnicodeDecodeError, LookupError):
                    html_content = raw_content.decode('utf-8', errors='replace')

                if 'html' in content_type.lower():
                    parser = HTMLTextExtractor()
                    parser.feed(html_content)
                    text_content = parser.get_text()
                else:
                    text_content = html_content

                if len(text_content) > max_chars:
                    text_content = text_content[:max_chars] + "\n\n[Content truncated]"
                elif was_truncated:
                    text_content += "\n\n[Content truncated]"

                title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else url

                self.console.print(f"[dim cyan]🌐 URL: {url} ({len(text_content)} chars)[/dim cyan]")
                return f"URL: {url}\nTitle: {title}\n\nContent:\n{text_content}"

        except Exception as e:
            self.console.print(f"[red]❌ Error fetching URL {url}: {e}[/red]")
            return None

    def _expand_git_content(self, git_ref: str, max_chars: int) -> str | None:
        """Expand a git reference (diff, commit, log, etc.)."""
        try:
            git_ref_lower = git_ref.lower()
            cmd, label = self._get_git_command(git_ref, git_ref_lower)

            self.console.print(f"[dim]🔀 Running: {' '.join(cmd)}...[/dim]")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown git error"
                self.console.print(f"[yellow]⚠️  Git command failed: {error_msg}[/yellow]")
                return None

            content = result.stdout.strip()

            if not content:
                self.console.print("[yellow]⚠️  Git command returned no output[/yellow]")
                return f"{label}\n\n(No output - working directory may be clean)"

            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Output truncated]"

            self.console.print(f"[dim cyan]🔀 Git: {git_ref} ({len(content)} chars)[/dim cyan]")
            return f"{label}\n```diff\n{content}\n```"

        except subprocess.TimeoutExpired:
            self.console.print("[red]❌ Git command timed out[/red]")
            return None
        except FileNotFoundError:
            self.console.print("[red]❌ Git not found - is git installed?[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]❌ Git error: {e}[/red]")
            return None

    def _get_git_command(self, git_ref: str, git_ref_lower: str) -> tuple[list[str], str]:
        """Get git command and label for a reference."""
        if git_ref_lower == 'diff':
            return ['git', 'diff'], "Git Diff (unstaged changes)"
        elif git_ref_lower in ('staged', 'cached'):
            return ['git', 'diff', '--cached'], "Git Diff (staged changes)"
        elif git_ref_lower == 'status':
            return ['git', 'status'], "Git Status"
        elif git_ref_lower == 'log':
            return ['git', 'log', '--oneline', '-20'], "Git Log (last 20 commits)"
        elif git_ref_lower in ('branch', 'branches'):
            return ['git', 'branch', '-a'], "Git Branches"
        elif git_ref_lower.startswith('diff:'):
            ref = git_ref[5:]
            return ['git', 'diff', ref], f"Git Diff ({ref})"
        elif git_ref_lower.startswith('show:'):
            ref = git_ref[5:]
            return ['git', 'show', ref], f"Git Show ({ref})"
        elif git_ref_lower.startswith('log:'):
            ref = git_ref[4:]
            return ['git', 'log', '--oneline', '-10', '--', ref], f"Git Log ({ref})"
        elif git_ref.startswith('HEAD') or re.match(r'^[a-f0-9]{6,40}$', git_ref):
            return ['git', 'show', git_ref], f"Git Commit ({git_ref})"
        else:
            return ['git', 'diff', git_ref], f"Git Diff (from {git_ref})"

    def _read_file_content(self, abs_path: str, display_path: str) -> str | None:
        """Read file content with encoding fallback.

        Args:
            abs_path: Absolute path to file
            display_path: Path to show in messages

        Returns:
            File content or None
        """
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp949', 'euc-kr']

        for enc in encodings_to_try:
            try:
                with open(abs_path, encoding=enc, errors='strict') as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort with replacement
        try:
            with open(abs_path, encoding='utf-8', errors='replace') as f:
                content = f.read()
            self.console.print(f"[yellow]⚠️  {display_path} contains non-UTF-8 chars[/yellow]")
            return content
        except Exception as e:
            self.console.print(f"[red]❌ Cannot read {display_path}: {e}[/red]")
            return None


# Convenience instance for simple usage
_default_expander: ReferenceExpander | None = None


def get_expander(console: Console | None = None) -> ReferenceExpander:
    """Get or create a reference expander instance.

    Args:
        console: Optional console for output

    Returns:
        ReferenceExpander instance
    """
    global _default_expander
    if _default_expander is None or console is not None:
        _default_expander = ReferenceExpander(console=console)
    return _default_expander


def expand_file_references(user_input: str, console: Console | None = None) -> str:
    """Convenience function to expand @ references.

    Args:
        user_input: User input with @ references
        console: Optional console for output

    Returns:
        Expanded input with referenced content
    """
    return get_expander(console).expand_references(user_input)
