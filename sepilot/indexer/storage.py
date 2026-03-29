"""Index Storage - SQLite-based persistent storage for indexes."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .models import CallEdge, Dependency, IndexedFile, IndexedSymbol, IndexStatus

logger = logging.getLogger(__name__)


class IndexStorage:
    """SQLite-based storage for project indexes.

    Provides persistent storage for symbols, dependencies, and call graphs.
    """

    SCHEMA_VERSION = 1

    def __init__(self, storage_path: Path | str):
        """Initialize storage.

        Args:
            storage_path: Path to the SQLite database file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            # Check schema version
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            version = conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()

            if version is None or int(version[0]) < self.SCHEMA_VERSION:
                self._create_schema(conn)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create or update database schema."""
        # Symbols table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                column_start INTEGER DEFAULT 0,
                column_end INTEGER DEFAULT 0,
                signature TEXT,
                docstring TEXT,
                language TEXT DEFAULT 'unknown',
                visibility TEXT DEFAULT 'public',
                parent TEXT,
                is_async INTEGER DEFAULT 0,
                is_static INTEGER DEFAULT 0,
                generic_params TEXT,
                decorators TEXT,
                metadata TEXT
            )
        """)

        # Files table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                status TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                symbol_count INTEGER DEFAULT 0,
                import_count INTEGER DEFAULT 0,
                error_message TEXT
            )
        """)

        # Dependencies table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_file TEXT NOT NULL,
                to_module TEXT NOT NULL,
                import_type TEXT NOT NULL,
                symbols TEXT,
                alias TEXT,
                is_default INTEGER DEFAULT 0,
                is_star INTEGER DEFAULT 0,
                line_number INTEGER DEFAULT 0
            )
        """)

        # Call edges table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_file TEXT NOT NULL,
                caller_symbol TEXT NOT NULL,
                callee_symbol TEXT NOT NULL,
                callee_file TEXT,
                line_number INTEGER DEFAULT 0,
                call_count INTEGER DEFAULT 1
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deps_from ON dependencies(from_file)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deps_to ON dependencies(to_module)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_caller ON call_edges(caller_file, caller_symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_callee ON call_edges(callee_symbol)")

        # Update schema version
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            (str(self.SCHEMA_VERSION),),
        )

        conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.storage_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # Symbol operations

    def save_symbol(self, symbol: IndexedSymbol) -> int:
        """Save a symbol to storage.

        Args:
            symbol: The symbol to save

        Returns:
            The symbol's database ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO symbols (
                    name, kind, file_path, line_start, line_end,
                    column_start, column_end, signature, docstring,
                    language, visibility, parent, is_async, is_static,
                    generic_params, decorators, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.name,
                    symbol.kind,
                    symbol.file_path,
                    symbol.line_start,
                    symbol.line_end,
                    symbol.column_start,
                    symbol.column_end,
                    symbol.signature,
                    symbol.docstring,
                    symbol.language,
                    symbol.visibility,
                    symbol.parent,
                    1 if symbol.is_async else 0,
                    1 if symbol.is_static else 0,
                    json.dumps(symbol.generic_params),
                    json.dumps(symbol.decorators),
                    json.dumps(symbol.metadata),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def save_symbols(self, symbols: list[IndexedSymbol]) -> None:
        """Save multiple symbols efficiently.

        Args:
            symbols: List of symbols to save
        """
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO symbols (
                    name, kind, file_path, line_start, line_end,
                    column_start, column_end, signature, docstring,
                    language, visibility, parent, is_async, is_static,
                    generic_params, decorators, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.name,
                        s.kind,
                        s.file_path,
                        s.line_start,
                        s.line_end,
                        s.column_start,
                        s.column_end,
                        s.signature,
                        s.docstring,
                        s.language,
                        s.visibility,
                        s.parent,
                        1 if s.is_async else 0,
                        1 if s.is_static else 0,
                        json.dumps(s.generic_params),
                        json.dumps(s.decorators),
                        json.dumps(s.metadata),
                    )
                    for s in symbols
                ],
            )
            conn.commit()

    def get_symbols_by_name(self, name: str) -> list[IndexedSymbol]:
        """Get all symbols with a given name.

        Args:
            name: Symbol name

        Returns:
            List of matching symbols
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE name = ?", (name,)
            ).fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def get_symbols_by_file(self, file_path: str) -> list[IndexedSymbol]:
        """Get all symbols in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of symbols
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE file_path = ?", (file_path,)
            ).fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def search_symbols(
        self, query: str, kind: str | None = None, limit: int = 100
    ) -> list[IndexedSymbol]:
        """Search for symbols matching a query.

        Args:
            query: Search query
            kind: Optional filter by kind
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        with self._get_connection() as conn:
            sql = "SELECT * FROM symbols WHERE name LIKE ?"
            params: list = [f"%{query}%"]

            if kind:
                sql += " AND kind = ?"
                params.append(kind)

            sql += " LIMIT ?"
            params.append(int(limit))

            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_symbol(row) for row in rows]

    def delete_symbols_by_file(self, file_path: str) -> int:
        """Delete all symbols from a file.

        Args:
            file_path: Path to the file

        Returns:
            Number of deleted symbols
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM symbols WHERE file_path = ?", (file_path,)
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_symbol(self, row: sqlite3.Row) -> IndexedSymbol:
        """Convert a database row to an IndexedSymbol."""
        return IndexedSymbol(
            name=row["name"],
            kind=row["kind"],
            file_path=row["file_path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            column_start=row["column_start"],
            column_end=row["column_end"],
            signature=row["signature"],
            docstring=row["docstring"],
            language=row["language"],
            visibility=row["visibility"],
            parent=row["parent"],
            is_async=bool(row["is_async"]),
            is_static=bool(row["is_static"]),
            generic_params=json.loads(row["generic_params"] or "[]"),
            decorators=json.loads(row["decorators"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # File operations

    def save_file(self, file_info: IndexedFile) -> None:
        """Save file indexing info.

        Args:
            file_info: The file info to save
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO files (
                    file_path, language, status, indexed_at,
                    file_hash, symbol_count, import_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_info.file_path,
                    file_info.language,
                    file_info.status.value,
                    file_info.indexed_at.isoformat(),
                    file_info.file_hash,
                    file_info.symbol_count,
                    file_info.import_count,
                    file_info.error_message,
                ),
            )
            conn.commit()

    def get_file(self, file_path: str) -> IndexedFile | None:
        """Get file info.

        Args:
            file_path: Path to the file

        Returns:
            File info or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE file_path = ?", (file_path,)
            ).fetchone()

            if row:
                return IndexedFile(
                    file_path=row["file_path"],
                    language=row["language"],
                    status=IndexStatus(row["status"]),
                    indexed_at=datetime.fromisoformat(row["indexed_at"]),
                    file_hash=row["file_hash"],
                    symbol_count=row["symbol_count"],
                    import_count=row["import_count"],
                    error_message=row["error_message"],
                )
            return None

    def get_all_files(self) -> list[IndexedFile]:
        """Get all indexed files.

        Returns:
            List of file info
        """
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM files").fetchall()
            return [
                IndexedFile(
                    file_path=row["file_path"],
                    language=row["language"],
                    status=IndexStatus(row["status"]),
                    indexed_at=datetime.fromisoformat(row["indexed_at"]),
                    file_hash=row["file_hash"],
                    symbol_count=row["symbol_count"],
                    import_count=row["import_count"],
                    error_message=row["error_message"],
                )
                for row in rows
            ]

    def delete_file(self, file_path: str) -> None:
        """Delete a file and its associated data.

        Args:
            file_path: Path to the file
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM dependencies WHERE from_file = ?", (file_path,))
            conn.execute("DELETE FROM call_edges WHERE caller_file = ?", (file_path,))
            conn.commit()

    # Dependency operations

    def save_dependencies(self, dependencies: list[Dependency]) -> None:
        """Save multiple dependencies.

        Args:
            dependencies: List of dependencies to save
        """
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO dependencies (
                    from_file, to_module, import_type, symbols,
                    alias, is_default, is_star, line_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        d.from_file,
                        d.to_module,
                        d.import_type,
                        json.dumps(d.symbols),
                        d.alias,
                        1 if d.is_default else 0,
                        1 if d.is_star else 0,
                        d.line_number,
                    )
                    for d in dependencies
                ],
            )
            conn.commit()

    def get_dependencies(self, from_file: str) -> list[Dependency]:
        """Get dependencies of a file.

        Args:
            from_file: Source file path

        Returns:
            List of dependencies
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dependencies WHERE from_file = ?", (from_file,)
            ).fetchall()
            return [self._row_to_dependency(row) for row in rows]

    def get_dependents(self, module: str) -> list[str]:
        """Get files that depend on a module.

        Args:
            module: Module name

        Returns:
            List of file paths
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT from_file FROM dependencies WHERE to_module = ?",
                (module,),
            ).fetchall()
            return [row["from_file"] for row in rows]

    def _row_to_dependency(self, row: sqlite3.Row) -> Dependency:
        """Convert a database row to a Dependency."""
        return Dependency(
            from_file=row["from_file"],
            to_module=row["to_module"],
            import_type=row["import_type"],
            symbols=json.loads(row["symbols"] or "[]"),
            alias=row["alias"],
            is_default=bool(row["is_default"]),
            is_star=bool(row["is_star"]),
            line_number=row["line_number"],
        )

    # Call edge operations

    def save_call_edges(self, edges: list[CallEdge]) -> None:
        """Save multiple call edges.

        Args:
            edges: List of call edges to save
        """
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO call_edges (
                    caller_file, caller_symbol, callee_symbol,
                    callee_file, line_number, call_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.caller_file,
                        e.caller_symbol,
                        e.callee_symbol,
                        e.callee_file,
                        e.line_number,
                        e.call_count,
                    )
                    for e in edges
                ],
            )
            conn.commit()

    def get_outgoing_calls(self, file_path: str, symbol: str) -> list[CallEdge]:
        """Get outgoing calls from a symbol.

        Args:
            file_path: Caller file path
            symbol: Caller symbol name

        Returns:
            List of call edges
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM call_edges
                WHERE caller_file = ? AND caller_symbol = ?
                """,
                (file_path, symbol),
            ).fetchall()
            return [self._row_to_call_edge(row) for row in rows]

    def get_incoming_calls(self, symbol: str) -> list[CallEdge]:
        """Get incoming calls to a symbol.

        Args:
            symbol: Callee symbol name

        Returns:
            List of call edges
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM call_edges WHERE callee_symbol = ?", (symbol,)
            ).fetchall()
            return [self._row_to_call_edge(row) for row in rows]

    def _row_to_call_edge(self, row: sqlite3.Row) -> CallEdge:
        """Convert a database row to a CallEdge."""
        return CallEdge(
            caller_file=row["caller_file"],
            caller_symbol=row["caller_symbol"],
            callee_symbol=row["callee_symbol"],
            callee_file=row["callee_file"],
            line_number=row["line_number"],
            call_count=row["call_count"],
        )

    # Utility methods

    def get_statistics(self) -> dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            stats = {}
            stats["symbols"] = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            stats["files"] = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            stats["dependencies"] = conn.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
            stats["call_edges"] = conn.execute("SELECT COUNT(*) FROM call_edges").fetchone()[0]
            return stats

    def clear_all(self) -> None:
        """Clear all data from storage."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM symbols")
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM dependencies")
            conn.execute("DELETE FROM call_edges")
            conn.commit()


def get_storage_for_project(project_root: str) -> IndexStorage:
    """Get storage for a project.

    Args:
        project_root: Project root directory

    Returns:
        IndexStorage instance
    """
    # Create hash of project path for storage directory
    project_hash = hashlib.md5(project_root.encode(), usedforsecurity=False).hexdigest()[:12]
    storage_dir = Path.home() / ".sepilot" / "indexes" / project_hash
    return IndexStorage(storage_dir / "index.db")
