"""File reference completer for interactive mode.

Provides auto-completion for @file references with fuzzy matching.
"""

import os

from sepilot.ui.command_catalog import iter_completion_commands

try:
    from prompt_toolkit.completion import Completer, Completion
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    Completer = object


class FileReferenceCompleter(Completer):
    """Custom completer for @file references with auto-completion (Claude Code style)"""

    def __init__(self, working_dir: str = "."):
        self.working_dir = os.path.abspath(working_dir)
        self._file_cache = []
        self._cache_timestamp = 0
        self._cache_ttl = 5  # Refresh cache every 5 seconds
        self._gitignore_patterns = None
        self._gitignore_timestamp = 0
        self._max_files = int(os.getenv("SEPILOT_COMPLETER_MAX_FILES", "10000"))
        self._max_dirs = int(os.getenv("SEPILOT_COMPLETER_MAX_DIRS", "1000"))
        self._cache_limit_notice: str | None = None

        # Pre-compiled common ignore directory names for fast lookup
        self._ignored_dirs = {
            '.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env',
            'dist', 'build', '.pytest_cache', '.mypy_cache',
            '.tox', '.eggs', 'target', 'out', '.gradle', '.idea',
            '.vscode', '.next', '.nuxt', 'coverage', '.nyc_output'
        }

    def _load_gitignore_patterns(self):
        """Load and parse .gitignore patterns"""
        import time

        current_time = time.time()
        # Refresh gitignore cache if expired
        if current_time - self._gitignore_timestamp < self._cache_ttl and self._gitignore_patterns is not None:
            return self._gitignore_patterns

        patterns = []
        gitignore_path = os.path.join(self.working_dir, '.gitignore')

        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception:
                pass  # Ignore errors reading .gitignore

        self._gitignore_patterns = patterns
        self._gitignore_timestamp = current_time
        return patterns

    def _is_ignored(self, path: str) -> bool:
        """Check if a path matches any gitignore pattern (optimized, with negation)"""
        # Fast check: common ignored directories (set lookup is O(1))
        basename = os.path.basename(path)
        if basename in self._ignored_dirs:
            return True

        # Fast check: common patterns
        if basename.endswith(('.pyc', '.pyo', '.so', '.dylib', '.egg-info')):
            return True

        # Check if any path component is an ignored directory
        for component in path.split(os.sep):
            if component in self._ignored_dirs:
                return True

        # Only load and check gitignore patterns if we haven't matched yet
        patterns = self._load_gitignore_patterns()
        if not patterns:
            return False

        import fnmatch

        # Check .gitignore patterns (respect order and negation)
        ignore = False
        for pattern in patterns:
            # Skip empty patterns
            if not pattern:
                continue

            # Handle directory patterns (ending with /)
            if pattern.endswith('/'):
                pattern = pattern[:-1]
                if basename == pattern or path.startswith(pattern + os.sep):
                    ignore = True
                    continue
            # Handle negation patterns (starting with !)
            elif pattern.startswith('!'):
                neg_pattern = pattern[1:]
                # Un-ignore if matched previously
                if fnmatch.fnmatch(basename, neg_pattern) or fnmatch.fnmatch(path, neg_pattern) or (
                    '/' in neg_pattern and fnmatch.fnmatch(path, '*/' + neg_pattern)
                ):
                    ignore = False
                continue
            # Handle glob patterns
            else:
                # Check basename match first (faster)
                if fnmatch.fnmatch(basename, pattern):
                    ignore = True
                    continue
                # Check full path match
                if fnmatch.fnmatch(path, pattern):
                    ignore = True
                    continue
                # Check if pattern matches with path separator
                if '/' in pattern and fnmatch.fnmatch(path, '*/' + pattern):
                    ignore = True
                    continue

        return ignore

    def _build_file_cache(self):
        """Build a cache of all files in the codebase for fast fuzzy matching"""
        import time

        current_time = time.time()
        # Refresh cache if expired
        if current_time - self._cache_timestamp < self._cache_ttl and self._file_cache:
            return self._file_cache

        files = []
        file_count = 0
        file_limit_hit = False
        try:
            # Walk through entire directory tree
            for root, dirs, filenames in os.walk(self.working_dir):
                # Early exit if we've collected enough files
                if file_count >= self._max_files:
                    file_limit_hit = True
                    break

                # Fast filter: remove ignored directories using set lookup
                dirs[:] = [d for d in dirs if d not in self._ignored_dirs]

                # Only check gitignore for remaining directories
                if dirs:
                    filtered_dirs = []
                    for d in dirs:
                        # Check relative path only if needed
                        rel_path = os.path.relpath(os.path.join(root, d), self.working_dir)
                        if not self._is_ignored(rel_path):
                            filtered_dirs.append(d)
                    dirs[:] = filtered_dirs

                # Process files in this directory
                for filename in filenames:
                    # Early exit check
                    if file_count >= self._max_files:
                        file_limit_hit = True
                        break

                    # Skip common ignored file extensions
                    if filename.endswith(('.pyc', '.pyo', '.so', '.dylib')):
                        continue

                    rel_path = os.path.relpath(os.path.join(root, filename), self.working_dir)

                    # Check if file should be ignored
                    if self._is_ignored(rel_path):
                        continue

                    files.append({
                        'path': rel_path,
                        'name': filename,
                        'is_dir': False
                    })
                    file_count += 1

            # Add directories for navigation (limited)
            dir_count = 0
            dir_limit_hit = False
            for root, dirs, _ in os.walk(self.working_dir):
                if dir_count >= self._max_dirs:
                    dir_limit_hit = True
                    break

                # Fast filter: remove ignored directories
                dirs[:] = [d for d in dirs if d not in self._ignored_dirs]

                if dirs:
                    filtered_dirs = []
                    for d in dirs:
                        rel_path = os.path.relpath(os.path.join(root, d), self.working_dir)
                        if not self._is_ignored(rel_path):
                            filtered_dirs.append(d)
                    dirs[:] = filtered_dirs

                for dirname in dirs:
                    if dir_count >= self._max_dirs:
                        dir_limit_hit = True
                        break

                    rel_path = os.path.relpath(os.path.join(root, dirname), self.working_dir)
                    files.append({
                        'path': rel_path,
                        'name': dirname,
                        'is_dir': True
                    })
                    dir_count += 1

        except (OSError, PermissionError):
            pass

        if file_limit_hit or dir_limit_hit:
            notices = []
            if file_limit_hit:
                notices.append(f"{self._max_files:,} files")
            if dir_limit_hit:
                notices.append(f"{self._max_dirs:,} dirs")
            joined = ", ".join(notices)
            self._cache_limit_notice = (
                f"Index capped at {joined}; refine the path or raise SEPILOT_COMPLETER_MAX_*"
            )
        else:
            self._cache_limit_notice = None

        self._file_cache = files
        self._cache_timestamp = current_time
        return files

    def _fuzzy_match_score(self, query: str, path: str) -> int:
        """Calculate fuzzy match score (higher = better match)

        Scoring:
        - Exact filename match: 1000
        - Filename starts with query: 500
        - Filename contains query: 300
        - Path contains query (case insensitive): 100
        - Query chars appear in order: 50
        """
        query_lower = query.lower()
        path_lower = path.lower()
        filename = os.path.basename(path_lower)

        score = 0

        # Exact match
        if filename == query_lower:
            score += 1000

        # Filename starts with query
        if filename.startswith(query_lower):
            score += 500

        # Filename contains query
        if query_lower in filename:
            score += 300

        # Path contains query
        if query_lower in path_lower:
            score += 100

        # Sequential character match (fuzzy)
        query_idx = 0
        for char in path_lower:
            if query_idx < len(query_lower) and char == query_lower[query_idx]:
                query_idx += 1

        if query_idx == len(query_lower):  # All chars matched in order
            score += 50

        # Prefer shorter paths (less nested)
        depth_penalty = path.count('/') * 10
        score -= depth_penalty

        return score

    def _get_file_completions(self, partial_path: str):
        """Get file completions using fuzzy matching across entire codebase

        Args:
            partial_path: Partial file path after @

        Yields:
            Completion objects for matching files (sorted by relevance)
        """
        if not HAS_PROMPT_TOOLKIT:
            return

        # Build/refresh file cache
        all_files = self._build_file_cache()

        # If empty query, show files in current directory only
        if not partial_path:
            matches = []
            for file_info in all_files:
                path = file_info['path']
                # Only show top-level files/dirs
                if '/' not in path and not path.startswith('.'):
                    matches.append((0, file_info))

            # Sort by name
            matches.sort(key=lambda x: x[1]['name'])

            for _, file_info in matches[:50]:  # Limit to 50 results
                path = file_info['path']
                is_dir = file_info['is_dir']

                display_text = path + ('/' if is_dir else '')
                completion_text = path + ('/' if is_dir else '')

                yield Completion(
                    text=completion_text,
                    start_position=0,
                    display=display_text,
                    display_meta='dir' if is_dir else 'file'
                )
            if self._cache_limit_notice:
                yield Completion(
                    text="",
                    start_position=0,
                    display="[index limited]",
                    display_meta=self._cache_limit_notice[:70],
                )
            return

        # Fuzzy match across all files
        matches = []
        for file_info in all_files:
            path = file_info['path']
            score = self._fuzzy_match_score(partial_path, path)

            # Only include if there's some match
            if score > 0:
                matches.append((score, file_info))

        # Sort by score (descending) - best matches first
        matches.sort(key=lambda x: x[0], reverse=True)

        # Yield completions (limit to top 50)
        for score, file_info in matches[:50]:
            path = file_info['path']
            is_dir = file_info['is_dir']

            # Display shows the full path
            display_text = path + ('/' if is_dir else '')

            # Completion replaces the entire partial_path
            completion_text = path + ('/' if is_dir else '')
            start_pos = -len(partial_path)

            yield Completion(
                text=completion_text,
                start_position=start_pos,
                display=display_text,
                display_meta=f'dir ({score})' if is_dir else f'file ({score})'
            )

        if self._cache_limit_notice:
            yield Completion(
                text=partial_path,
                start_position=-len(partial_path),
                display='[index limited]',
                display_meta=self._cache_limit_notice[:70],
            )

    def get_completions(self, document, complete_event):
        """Provide completions for @ references and /commands (Claude Code style)

        Supports:
            @file.py      - File path completions
            @folder/      - Directory completions
            @url:         - URL reference hint
            @git:         - Git reference completions
            @**/*.py      - Glob pattern hint
            /command      - Command completions
        """
        if not HAS_PROMPT_TOOLKIT:
            return

        text = document.text_before_cursor

        # Check for / command completion at start of line
        if text.startswith('/') and ' ' not in text:
            for completion in self._get_command_completions(text[1:]):
                yield completion
            return

        # Find the last @ symbol position
        last_at_pos = text.rfind('@')

        # No @ found, no completions
        if last_at_pos == -1:
            return

        # Get text after the last @
        after_at = text[last_at_pos + 1:]

        # If there's a space in after_at, we've moved past this @ reference
        if ' ' in after_at:
            return

        # Check for special reference types
        if after_at == '' or after_at in 'url:git:':
            # Show all reference type hints
            for completion in self._get_reference_type_completions(after_at):
                yield completion
            # Also show file completions
            for completion in self._get_file_completions(after_at):
                yield completion
            return

        # @url: completions
        if after_at.startswith('url:'):
            # Show URL hint
            url_part = after_at[4:]
            if not url_part:
                yield Completion(
                    text='url:https://',
                    start_position=-len(after_at),
                    display='@url:https://...',
                    display_meta='Web page content'
                )
            return

        # @git: completions
        if after_at.startswith('git:'):
            for completion in self._get_git_completions(after_at[4:]):
                yield completion
            return

        # @agent: completions
        if after_at.startswith('agent:'):
            for completion in self._get_agent_completions(after_at[6:]):
                yield completion
            return

        # Get completions for the partial path (files and folders)
        for completion in self._get_file_completions(after_at):
            yield completion

    def _get_agent_completions(self, partial: str):
        """Get completions for @agent: references"""
        if not HAS_PROMPT_TOOLKIT:
            return

        agents = [
            ('explore', 'Codebase exploration and analysis'),
            ('coder', 'Code generation and implementation'),
            ('reviewer', 'Code review and suggestions'),
            ('refactor', 'Code refactoring'),
            ('docs', 'Documentation generation'),
            ('test', 'Test generation'),
            ('debug', 'Debugging assistance'),
        ]

        partial_lower = partial.lower()
        for agent, description in agents:
            if agent.startswith(partial_lower) or partial_lower in agent:
                yield Completion(
                    text=f'agent:{agent}',
                    start_position=-len(partial) - 6,  # -6 for 'agent:'
                    display=f'@agent:{agent}',
                    display_meta=description
                )

    def _get_reference_type_completions(self, partial: str):
        """Get completions for reference types (@url:, @git:, @agent:, etc.)"""
        if not HAS_PROMPT_TOOLKIT:
            return

        reference_types = [
            ('url:', 'Web page content (@url:https://...)'),
            ('git:diff', 'Unstaged changes'),
            ('git:staged', 'Staged changes'),
            ('git:status', 'Git status'),
            ('git:log', 'Recent commits'),
            ('git:branch', 'List branches'),
            ('agent:explore', 'Codebase exploration'),
            ('agent:coder', 'Code generation'),
            ('agent:reviewer', 'Code review'),
            ('agent:refactor', 'Refactoring'),
            ('agent:docs', 'Documentation'),
            ('agent:test', 'Test generation'),
            ('agent:debug', 'Debugging'),
            ('**/*.py', 'Glob pattern (all .py files)'),
        ]

        for ref_type, description in reference_types:
            if ref_type.startswith(partial.lower()):
                yield Completion(
                    text=ref_type,
                    start_position=-len(partial),
                    display=f'@{ref_type}',
                    display_meta=description
                )

    def _get_git_completions(self, partial: str):
        """Get completions for @git: references"""
        if not HAS_PROMPT_TOOLKIT:
            return

        git_refs = [
            ('diff', 'Unstaged changes'),
            ('staged', 'Staged changes'),
            ('status', 'Git status'),
            ('log', 'Recent commits (last 20)'),
            ('branch', 'List all branches'),
            ('HEAD', 'Current commit'),
            ('HEAD~1', 'Previous commit'),
            ('HEAD~3', 'Last 3 commits'),
            ('diff:main', 'Diff from main branch'),
            ('diff:HEAD~1', 'Diff from previous commit'),
            ('show:', 'Show specific commit'),
            ('log:', 'Log for specific file'),
        ]

        partial_lower = partial.lower()
        for ref, description in git_refs:
            if ref.lower().startswith(partial_lower) or partial_lower in ref.lower():
                yield Completion(
                    text=f'git:{ref}',
                    start_position=-len(partial) - 4,  # -4 for 'git:'
                    display=f'@git:{ref}',
                    display_meta=description
                )

    def _get_command_completions(self, partial: str):
        """Get completions for / commands (Claude Code style)"""
        if not HAS_PROMPT_TOOLKIT:
            return

        partial_lower = partial.lower()

        builtin_commands = [
            (entry.name.lstrip('/'), entry.description)
            for entry in iter_completion_commands()
        ]

        # Get custom commands
        try:
            from sepilot.commands import get_command_manager
            cmd_manager = get_command_manager()
            custom_cmds = [(c.name, c.description) for c in cmd_manager.list_commands()]
        except Exception:
            custom_cmds = []

        # Get skills
        try:
            from sepilot.skills import get_skill_manager
            skill_manager = get_skill_manager()
            skill_cmds = [(f"skill {s.name}", s.description) for s in skill_manager.list_skills()]
        except Exception:
            skill_cmds = []

        all_commands = builtin_commands + custom_cmds + skill_cmds

        # Filter and sort by relevance
        matches = []
        for cmd, desc in all_commands:
            if cmd.startswith(partial_lower) or partial_lower in cmd:
                # Score: exact prefix match > contains
                score = 100 if cmd.startswith(partial_lower) else 50
                matches.append((cmd, desc, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)

        for cmd, desc, _score in matches[:20]:  # Limit to 20 suggestions
            yield Completion(
                text=cmd,
                start_position=-len(partial),
                display=f"/{cmd}",
                display_meta=desc[:50]
            )
