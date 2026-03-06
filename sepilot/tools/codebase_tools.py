"""Codebase exploration and navigation tools

Inspired by Claude Code's efficient search strategies:
- Incremental reading: Read files in chunks, not all at once
- Smart file prioritization: Focus on relevant files first
- Early termination: Stop when we have enough information
- Better caching: Cache file metadata, not just content
- Parallel search when possible
- Use ripgrep for fast content search
"""

import contextlib
import fnmatch
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sepilot.tools.base_tool import BaseTool

# Optimized limits based on Claude Code's approach
MAX_FILE_SIZE_FOR_FULL_READ = 100_000  # 100KB (reduced from 1MB)
CHUNK_SIZE = 4096  # Read in 4KB chunks
MAX_LINES_PER_CHUNK = 50  # Process 50 lines at a time
MAX_SEARCH_DEPTH = 5  # Maximum directory depth for exploration
MAX_FILES_TO_SCAN = 100  # Stop after scanning this many files
CACHE_TTL = 300  # Cache for 5 minutes


@dataclass
class FileMetadata:
    """Cached file metadata to avoid repeated file system calls"""
    path: str
    size: int
    modified_time: float
    is_binary: bool
    extension: str
    depth: int  # Directory depth from project root

    def __hash__(self):
        return hash(self.path)


class IncrementalFileReader:
    """Read files incrementally in chunks for better performance"""

    def __init__(self, file_path: Path, chunk_size: int = CHUNK_SIZE):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._file = None
        self._line_buffer = []
        self._line_number = 0

    def __enter__(self):
        self._file = open(self.file_path, encoding='utf-8', errors='ignore')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def read_lines(self, max_lines: int = MAX_LINES_PER_CHUNK) -> Iterator[tuple[int, str]]:
        """Read up to max_lines from the file"""
        lines_read = 0

        while lines_read < max_lines:
            chunk = self._file.read(self.chunk_size)
            if not chunk:
                # End of file, yield remaining buffer
                if self._line_buffer:
                    self._line_number += 1
                    yield self._line_number, ''.join(self._line_buffer)
                break

            # Process chunk
            lines = chunk.split('\n')

            # First line continues from buffer
            if self._line_buffer:
                self._line_buffer.append(lines[0])
                if len(lines) > 1:
                    # We have a complete line
                    self._line_number += 1
                    yield self._line_number, ''.join(self._line_buffer)
                    self._line_buffer = []
                    lines_read += 1

            # Process middle lines
            for line in lines[1:-1]:
                if lines_read >= max_lines:
                    break
                self._line_number += 1
                yield self._line_number, line
                lines_read += 1

            # Last line goes to buffer (may be incomplete)
            if lines[-1]:
                self._line_buffer = [lines[-1]]

    def search_pattern(self, pattern: str, max_matches: int = 5) -> list[tuple[int, str]]:
        """Search for a pattern in the file incrementally"""
        matches = []
        pattern_lower = pattern.lower()

        for line_no, line in self.read_lines(max_lines=500):  # Scan first 500 lines
            if pattern_lower in line.lower():
                matches.append((line_no, line.strip()))
                if len(matches) >= max_matches:
                    break

        return matches


class CodebaseExplorer:
    """Codebase explorer using Claude Code strategies"""

    def __init__(self, logger=None):
        self.logger = logger
        self.project_root = self._find_project_root()

        # Enhanced caching
        self._file_metadata_cache: dict[str, FileMetadata] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._directory_cache: dict[str, set[str]] = {}

        # Gitignore patterns
        self._gitignore_patterns = self._load_gitignore()

        # Check for ripgrep or grep
        self._ripgrep_path = shutil.which('rg')
        self._has_ripgrep = self._ripgrep_path is not None
        self._grep_path = shutil.which('grep')
        self._has_grep = self._grep_path is not None

        # Priority patterns for file exploration
        self._priority_patterns = {
            'high': ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.go', '*.rs'],
            'medium': ['*.json', '*.yaml', '*.yml', '*.toml', '*.xml'],
            'low': ['*.md', '*.txt', '*.rst', '*.log']
        }

    def _find_project_root(self) -> Path:
        """Find the project root directory"""
        current = Path.cwd()
        markers = ['.git', 'pyproject.toml', 'setup.py', 'package.json', 'Cargo.toml', 'go.mod']

        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent

        return Path.cwd()

    def _load_gitignore(self) -> list[str]:
        """Load gitignore patterns"""
        patterns = [
            '__pycache__', '*.pyc', '*.pyo', '*.pyd', '.Python',
            'node_modules', '.git', '.venv', 'venv', 'env', '.env',
            '*.egg-info', 'dist', 'build', '.DS_Store', 'Thumbs.db',
            '.idea', '.vscode', '*.swp', '*.swo', '*~'
        ]

        gitignore_path = self.project_root / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception:
                pass

        return patterns

    def _should_ignore(self, path: Path) -> bool:
        """Quick check if path should be ignored"""
        name = path.name

        # Quick checks first
        if name.startswith('.') and name != '.gitignore':
            return True

        # Check patterns
        return any(fnmatch.fnmatch(name, pattern) for pattern in self._gitignore_patterns)

    def _is_binary_file(self, file_path: Path) -> bool:
        """Quick check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except Exception:
            return True

    def _get_file_metadata(self, file_path: Path) -> FileMetadata | None:
        """Get cached file metadata"""
        path_str = str(file_path)

        # Check cache
        if path_str in self._file_metadata_cache:
            if time.time() - self._cache_timestamps.get(path_str, 0) < CACHE_TTL:
                return self._file_metadata_cache[path_str]

        # Create new metadata
        try:
            stat = file_path.stat()
            rel_path = file_path.relative_to(self.project_root)
            depth = len(rel_path.parts)

            metadata = FileMetadata(
                path=str(rel_path),
                size=stat.st_size,
                modified_time=stat.st_mtime,
                is_binary=self._is_binary_file(file_path),
                extension=file_path.suffix,
                depth=depth
            )

            # Cache it
            self._file_metadata_cache[path_str] = metadata
            self._cache_timestamps[path_str] = time.time()

            return metadata
        except Exception:
            return None

    def find_files_incremental(self,
                              pattern: str = "*",
                              max_results: int = 50,
                              prioritize: bool = True) -> Iterator[str]:
        """Find files incrementally - yield results as we find them

        Claude Code strategy: Don't build full list first, yield as we go
        """
        results_found = 0
        visited_dirs = set()

        def search_directory(dir_path: Path, depth: int = 0):
            nonlocal results_found

            if depth > MAX_SEARCH_DEPTH:
                return

            if dir_path in visited_dirs:
                return
            visited_dirs.add(dir_path)

            try:
                entries = list(dir_path.iterdir())

                # Sort by priority if requested
                if prioritize:
                    entries.sort(key=lambda p: (
                        not p.is_file(),  # Files first
                        0 if p.suffix in ['.py', '.js', '.ts'] else 1,  # Priority extensions
                        p.name
                    ))

                for entry in entries:
                    if results_found >= max_results:
                        return

                    if self._should_ignore(entry):
                        continue

                    if entry.is_file():
                        if fnmatch.fnmatch(entry.name, pattern):
                            rel_path = entry.relative_to(self.project_root)
                            yield str(rel_path)
                            results_found += 1
                    elif entry.is_dir():
                        # Recursively search subdirectory
                        yield from search_directory(entry, depth + 1)

            except PermissionError:
                pass

        # Start search from project root
        yield from search_directory(self.project_root)

    def search_content_incremental(self,
                                  search_term: str,
                                  file_pattern: str = "*.py",
                                  max_results: int = 20) -> Iterator[tuple[str, list[tuple[int, str]]]]:
        """Search file contents incrementally

        Claude Code strategy: Stream results, don't accumulate everything first
        Uses ripgrep > grep > python fallback (in order of availability)
        """
        if self._has_ripgrep:
            yield from self._ripgrep_search_incremental(search_term, file_pattern, max_results)
        elif self._has_grep:
            yield from self._grep_search_incremental(search_term, file_pattern, max_results)
        else:
            yield from self._python_search_incremental(search_term, file_pattern, max_results)

    def _ripgrep_search_incremental(self,
                                   search_term: str,
                                   file_pattern: str,
                                   max_results: int) -> Iterator[tuple[str, list[tuple[int, str]]]]:
        """Use ripgrep for incremental search"""
        cmd = [
            self._ripgrep_path,
            '-i',  # case insensitive
            '-n',  # line numbers
            '--max-count', '3',  # Max 3 matches per file
            '--max-filesize', '500K',  # Skip files > 500KB (increased from 100K)
            '-g', file_pattern,
            search_term,
            str(self.project_root)
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=str(self.project_root)
            )

            current_file = None
            current_matches = []
            results_yielded = 0

            for line in process.stdout:
                if results_yielded >= max_results:
                    process.terminate()
                    break

                try:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_no = int(parts[1])
                        content = parts[2].strip()

                        if current_file != file_path:
                            # Yield previous file's results
                            if current_file and current_matches:
                                rel_path = str(Path(current_file).relative_to(self.project_root))
                                yield (rel_path, current_matches)
                                results_yielded += 1

                            current_file = file_path
                            current_matches = []

                        current_matches.append((line_no, content[:100]))  # Limit line length

                except Exception:
                    continue

            # Yield last file's results
            if current_file and current_matches and results_yielded < max_results:
                rel_path = str(Path(current_file).relative_to(self.project_root))
                yield (rel_path, current_matches)

        except Exception:
            pass

    def _grep_search_incremental(self,
                                 search_term: str,
                                 file_pattern: str,
                                 max_results: int) -> Iterator[tuple[str, list[tuple[int, str]]]]:
        """Use system grep as fallback when ripgrep is not available.

        grep -rin is universally available on Linux/macOS and handles
        large codebases much better than the Python-based search.
        """
        # Convert glob pattern to --include format for grep
        include_flag = f'--include={file_pattern}'
        cmd = [
            self._grep_path,
            '-r',  # recursive
            '-i',  # case insensitive
            '-n',  # line numbers
            '-F',  # fixed strings (literal match, no regex)
            '-m', '3',  # max 3 matches per file
            include_flag,
            search_term,
            str(self.project_root)
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=str(self.project_root)
            )

            current_file = None
            current_matches = []
            results_yielded = 0

            for line in process.stdout:
                if results_yielded >= max_results:
                    process.terminate()
                    break

                try:
                    # grep output: file_path:line_no:content
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_no = int(parts[1])
                        content = parts[2].strip()

                        if current_file != file_path:
                            if current_file and current_matches:
                                with contextlib.suppress(ValueError):
                                    rel_path = str(Path(current_file).relative_to(self.project_root))
                                    yield (rel_path, current_matches)
                                    results_yielded += 1

                            current_file = file_path
                            current_matches = []

                        current_matches.append((line_no, content[:100]))
                except Exception:
                    continue

            # Yield last file's results
            if current_file and current_matches and results_yielded < max_results:
                with contextlib.suppress(ValueError):
                    rel_path = str(Path(current_file).relative_to(self.project_root))
                    yield (rel_path, current_matches)

        except Exception:
            # If grep also fails, fall back to Python search
            yield from self._python_search_incremental(search_term, file_pattern, max_results)

    def _python_search_incremental(self,
                                  search_term: str,
                                  file_pattern: str,
                                  max_results: int) -> Iterator[tuple[str, list[tuple[int, str]]]]:
        """Python-based incremental search"""
        _search_lower = search_term.lower()  # noqa: F841
        results_yielded = 0

        # Use incremental file finding
        for file_path in self.find_files_incremental(file_pattern, max_results * 3):
            if results_yielded >= max_results:
                break

            full_path = self.project_root / file_path

            # Skip large or binary files
            metadata = self._get_file_metadata(full_path)
            if not metadata or metadata.is_binary or metadata.size > MAX_FILE_SIZE_FOR_FULL_READ:
                continue

            # Search in file incrementally
            try:
                with IncrementalFileReader(full_path) as reader:
                    matches = reader.search_pattern(search_term, max_matches=3)
                    if matches:
                        yield (file_path, matches)
                        results_yielded += 1
            except Exception:
                continue

    def get_smart_file_preview(self, file_path: str, context_lines: int = 20) -> dict[str, Any]:
        """Get smart preview of file - focus on important parts

        Claude Code strategy: Don't read everything, focus on:
        1. File header (imports, module docs)
        2. Class/function definitions
        3. Recent changes (if git available)
        """
        full_path = self.project_root / file_path

        if not full_path.exists():
            return {"error": f"File {file_path} not found"}

        metadata = self._get_file_metadata(full_path)
        if not metadata:
            return {"error": f"Cannot read metadata for {file_path}"}

        preview = {
            "path": file_path,
            "size": metadata.size,
            "extension": metadata.extension,
            "header": [],
            "structure": [],
            "preview_lines": []
        }

        # For huge files, only read beginning
        if metadata.size > MAX_FILE_SIZE_FOR_FULL_READ:
            preview["truncated"] = True

        try:
            with IncrementalFileReader(full_path) as reader:
                line_count = 0
                in_header = True

                for line_no, line in reader.read_lines(max_lines=context_lines * 2):
                    line_count += 1
                    stripped = line.strip()

                    # Collect header (imports, module docstring)
                    if in_header:
                        if stripped and not stripped.startswith(('#', 'import', 'from', '"""', "'''")):
                            in_header = False
                        else:
                            preview["header"].append(f"{line_no}: {line[:100]}")

                    # Find structure (classes, functions)
                    if stripped.startswith(('def ', 'class ', 'async def ')):
                        preview["structure"].append(f"{line_no}: {stripped[:80]}")

                    # Collect preview lines
                    if line_count <= context_lines:
                        preview["preview_lines"].append(f"{line_no}: {line[:100]}")

        except Exception as e:
            preview["error"] = str(e)

        return preview

    def analyze_project_structure_smart(self, max_depth: int = 3) -> dict[str, Any]:
        """Analyze project structure smartly - focus on important directories

        Claude Code strategy: Prioritize source code directories
        """
        structure = {
            "root": str(self.project_root),
            "summary": {},
            "key_directories": [],
            "total_files": 0,
            "languages": {}
        }

        # Priority directories to explore first
        priority_dirs = ['src', 'app', 'lib', 'components', 'pages', 'api', 'core', 'utils']

        def analyze_directory(dir_path: Path, depth: int = 0) -> dict[str, int]:
            if depth > max_depth:
                return {}

            stats = {"files": 0, "dirs": 0, "size": 0}

            try:
                for entry in dir_path.iterdir():
                    if self._should_ignore(entry):
                        continue

                    if entry.is_file():
                        stats["files"] += 1
                        try:
                            stats["size"] += entry.stat().st_size

                            # Track languages
                            ext = entry.suffix
                            if ext:
                                structure["languages"][ext] = structure["languages"].get(ext, 0) + 1
                        except Exception:
                            pass
                    elif entry.is_dir():
                        stats["dirs"] += 1

                        # Explore priority directories deeper
                        if entry.name in priority_dirs or depth < 2:
                            sub_stats = analyze_directory(entry, depth + 1)
                            stats["files"] += sub_stats.get("files", 0)
                            stats["size"] += sub_stats.get("size", 0)

            except PermissionError:
                pass

            return stats

        # Start analysis
        structure["summary"] = analyze_directory(self.project_root)
        structure["total_files"] = structure["summary"].get("files", 0)

        # Find key directories
        for name in priority_dirs:
            dir_path = self.project_root / name
            if dir_path.exists() and dir_path.is_dir():
                structure["key_directories"].append(name)

        return structure

    # ===========================================================================
    # Compatibility methods for existing CodebaseExplorer interface
    # ===========================================================================

    def find_file_by_name(self, filename: str) -> str | None:
        """Find a file by exact name (compatibility method)"""
        for file_path in self.find_files_incremental(f"*{filename}", max_results=1):
            return file_path
        return None

    def find_files_by_pattern(self, pattern: str, max_files: int = 100) -> list[str]:
        """Find files matching a pattern (compatibility method)"""
        return list(self.find_files_incremental(pattern, max_results=max_files))

    def get_all_files(self, max_files: int = 100) -> list[str]:
        """Get all files in project (compatibility method)"""
        return list(self.find_files_incremental("*", max_results=max_files))

    def find_files_with_content(self,
                                search_term: str,
                                file_pattern: str = "*.py",
                                max_results: int = 20) -> list[tuple[str, list[tuple[int, str]]]]:
        """Search files for content (compatibility method)"""
        return list(self.search_content_incremental(search_term, file_pattern, max_results))

    def find_class_or_function(self, name: str, max_results: int = 20) -> list[tuple[str, int]]:
        """Find class or function definitions (compatibility method)"""
        results = []

        # Search for class and function definitions
        patterns = [
            f"class {name}",
            f"def {name}",
            f"function {name}",  # JavaScript
            f"const {name} =",  # JavaScript arrow functions
        ]

        for pattern in patterns:
            for file_path, matches in self.search_content_incremental(pattern, "*", max_results=10):
                for line_no, _ in matches:
                    results.append((file_path, line_no))
                    if len(results) >= max_results:
                        return results

        return results

    def get_project_structure(self) -> dict[str, Any]:
        """Get a hierarchical view of the project structure (compatibility method)"""
        structure_info = self.analyze_project_structure_smart(max_depth=4)

        # Convert to format expected by old CodebaseExplorer interface
        return {
            "root": structure_info["root"],
            "structure": {
                "type": "directory",
                "summary": structure_info["summary"],
                "key_directories": structure_info["key_directories"]
            }
        }

    def get_file_context(self, file_path: str) -> dict[str, Any]:
        """Get context about a file (imports, functions, classes) - compatibility method"""
        preview = self.get_smart_file_preview(file_path, context_lines=50)

        if "error" in preview:
            return {"error": preview["error"]}

        # Extract structure information into the expected format
        context = {
            "path": file_path,
            "size": preview.get("size", 0),
            "extension": preview.get("extension", ""),
            "imports": [],
            "functions": [],
            "classes": []
        }

        # Parse structure lines to extract classes and functions
        for line in preview.get("structure", []):
            # Format is "line_no: def/class name..."
            if ": class " in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    line_no = parts[0].strip()
                    content = parts[1].strip()
                    # Extract class name
                    if content.startswith("class "):
                        class_name = content[6:].split("(")[0].split(":")[0].strip()
                        with contextlib.suppress(ValueError):
                            context["classes"].append({"name": class_name, "line": int(line_no)})
            elif ": def " in line or ": async def " in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    line_no = parts[0].strip()
                    content = parts[1].strip()
                    # Extract function name
                    if "def " in content:
                        func_part = content.split("def ", 1)[1]
                        func_name = func_part.split("(")[0].strip()
                        with contextlib.suppress(ValueError):
                            context["functions"].append({"name": func_name, "line": int(line_no)})

        # Parse header for imports
        for line in preview.get("header", []):
            content = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            if content.startswith(("import ", "from ")):
                context["imports"].append(content)

        return context

    def suggest_related_files(self, file_path: str) -> list[str]:
        """Suggest files related to the given file (compatibility method)"""
        related = []
        file_path_obj = Path(file_path)
        base_name = file_path_obj.stem

        # Look for test files
        test_patterns = [
            f"test_{base_name}*.py",
            f"{base_name}_test*.py",
        ]

        for pattern in test_patterns:
            matches = self.find_files_by_pattern(pattern, max_files=5)
            related.extend(matches)

        # Look for similarly named files
        similar_matches = self.find_files_by_pattern(f"*{base_name}*", max_files=10)
        related.extend(similar_matches)

        # Remove duplicates and the original file
        seen = set()
        unique_related = []
        for f in related:
            if f not in seen and f != file_path:
                seen.add(f)
                unique_related.append(f)

        return unique_related[:10]  # Limit to top 10


class CodebaseTool(BaseTool):
    """Codebase exploration tool using Claude Code strategies"""

    name = "codebase"
    description = """Codebase exploration tool using incremental search strategies.

    Key improvements:
    - Incremental file reading (chunks instead of full files)
    - Smart prioritization (focus on source files first)
    - Early termination (stop when enough results found)
    - Better caching (metadata, not just content)

    Parameters:
    - action: find_files, search_content, preview_file, analyze_structure
    - pattern: File pattern for find_files
    - search_term: Term to search for
    - file_path: Path to file for preview
    - max_results: Maximum number of results (default 20)
    """

    def __init__(self, logger=None):
        super().__init__(logger)
        self.explorer = CodebaseExplorer(logger)

    def execute(self, action: str = "find_files", **kwargs) -> str:
        """Execute optimized codebase action"""

        if action == "find_files":
            pattern = kwargs.get("pattern", "*")
            max_results = kwargs.get("max_results", 20)

            output = ["Finding files incrementally..."]
            count = 0

            for file_path in self.explorer.find_files_incremental(pattern, max_results):
                output.append(f"  {file_path}")
                count += 1

            output.insert(1, f"Found {count} files matching '{pattern}':")
            return "\n".join(output)

        elif action == "search_content":
            search_term = kwargs.get("search_term", "")
            file_pattern = kwargs.get("file_pattern", "*.py")
            max_results = kwargs.get("max_results", 10)

            output = [f"Searching for '{search_term}' incrementally..."]
            count = 0

            for file_path, matches in self.explorer.search_content_incremental(
                search_term, file_pattern, max_results
            ):
                output.append(f"\n{file_path}:")
                for line_no, content in matches[:3]:
                    output.append(f"  Line {line_no}: {content[:80]}")
                count += 1

            output.insert(1, f"Found in {count} files:")
            return "\n".join(output)

        elif action == "preview_file":
            file_path = kwargs.get("file_path", "")
            context_lines = kwargs.get("context_lines", 20)

            preview = self.explorer.get_smart_file_preview(file_path, context_lines)

            if "error" in preview:
                return preview["error"]

            output = [f"Smart preview of {file_path}:"]
            output.append(f"Size: {preview['size']} bytes")

            if preview.get("truncated"):
                output.append("(File truncated - showing partial content)")

            if preview["header"]:
                output.append("\nHeader/Imports:")
                output.extend(preview["header"][:10])

            if preview["structure"]:
                output.append("\nStructure (classes/functions):")
                output.extend(preview["structure"][:15])

            if preview["preview_lines"]:
                output.append("\nFirst lines:")
                output.extend(preview["preview_lines"][:context_lines])

            return "\n".join(output)

        elif action == "analyze_structure":
            max_depth = kwargs.get("max_depth", 3)

            structure = self.explorer.analyze_project_structure_smart(max_depth)

            output = ["Project structure analysis:"]
            output.append(f"Root: {structure['root']}")
            output.append(f"Total files: {structure['total_files']}")

            if structure["key_directories"]:
                output.append(f"Key directories: {', '.join(structure['key_directories'])}")

            if structure["languages"]:
                output.append("\nLanguages detected:")
                sorted_langs = sorted(structure["languages"].items(), key=lambda x: x[1], reverse=True)
                for ext, count in sorted_langs[:10]:
                    output.append(f"  {ext}: {count} files")

            return "\n".join(output)

        else:
            return f"Unknown action: {action}"
