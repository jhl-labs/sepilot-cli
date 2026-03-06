"""File-based logging system for SE Pilot"""

import atexit
import json
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from sepilot.utils.text import sanitize_text


class FileLogger:
    """Handles logging of all operations to JSON Lines format

    Features:
    - Immediate flush to disk for critical logs
    - atexit handler for cleanup
    - Exception-safe logging
    """

    def __init__(
        self,
        log_dir: Path,
        session_id: str | None = None,
        continue_session: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Use deque with maxlen to prevent unbounded memory growth
        max_log_entries = 5000

        if continue_session and session_id:
            # Continue from existing session
            self.session_id = session_id
            self.session_file = self.log_dir / f"session_{session_id}.jsonl"
            self.logs: deque = deque(self._load_session(), maxlen=max_log_entries)
        else:
            # Start new session
            self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_file = self.log_dir / f"session_{self.session_id}.jsonl"
            self.logs: deque = deque(maxlen=max_log_entries)

        self.start_time = datetime.now()

        # Thread safety for concurrent logging
        self._lock = threading.RLock()

        # Keep file handle open for better performance
        self.log_file = open(self.session_file, 'a', encoding='utf-8')

        # Register cleanup handler
        atexit.register(self._cleanup)

    def _cleanup(self):
        """Cleanup handler called on program exit"""
        try:
            if hasattr(self, 'log_file') and not self.log_file.closed:
                self.log_file.flush()
                os.fsync(self.log_file.fileno())
                self.log_file.close()
        except Exception as e:
            # Log cleanup errors to stderr (can't use logger during cleanup)
            import sys
            print(f"WARNING: Failed to cleanup logger: {e}", file=sys.stderr)

    def _load_session(self) -> list[dict[str, Any]]:
        """Load existing session logs"""
        logs = []
        if self.session_file.exists():
            with open(self.session_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return logs

    def _write_log(self, entry: dict[str, Any], critical: bool = False) -> None:
        """Write a single log entry (thread-safe)

        Args:
            entry: Log entry to write
            critical: If True, force immediate flush to disk (for errors, iterations, tool calls)
        """
        with self._lock:
            entry["timestamp"] = datetime.now().isoformat()
            entry["session_id"] = self.session_id

            self.logs.append(entry)

            try:
                payload = json.dumps(entry, ensure_ascii=False)
                payload = sanitize_text(payload)
                self.log_file.write(payload + '\n')

                # Critical logs are flushed immediately
                if critical or entry.get("type") in ["error", "tool_call", "iteration"]:
                    self.log_file.flush()
                    os.fsync(self.log_file.fileno())
            except Exception as e:
                # If writing fails, try to write error to stderr
                import sys
                print(f"ERROR: Failed to write log: {e}", file=sys.stderr)

    def log_prompt(self, prompt: str) -> None:
        """Log the initial prompt"""
        self._write_log({
            "type": "prompt",
            "data": {
                "prompt": prompt
            }
        })

    def log_thought(self, thought: str) -> None:
        """Log agent's thought"""
        self._write_log({
            "type": "thought",
            "data": {
                "thought": thought
            }
        })

    def log_tool_call(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        output: Any,
        duration_ms: int | None = None
    ) -> None:
        """Log tool execution (critical - always flushed)"""
        self._write_log({
            "type": "tool_call",
            "data": {
                "tool_name": tool_name,
                "input": input_data,
                "output": str(output)[:5000],  # Truncate long outputs
                "duration_ms": duration_ms,
                "success": not str(output).startswith("Error:")
            }
        }, critical=True)

    def log_agent_state(self, state: dict[str, Any]) -> None:
        """Log agent state for debugging"""
        self._write_log({
            "type": "agent_state",
            "data": state
        })

    def log_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log application events (thread management, etc)."""
        self._write_log({
            "type": "event",
            "event_type": event_type,
            "data": details
        })

    def log_trace(self, trace_type: str, details: dict[str, Any]) -> None:
        """Log execution trace for detailed tracking"""
        self._write_log({
            "type": "trace",
            "trace_type": trace_type,
            "data": details
        })

    def log_error(self, error: str, context: dict | None = None) -> None:
        """Log errors (critical - always flushed)"""
        self._write_log({
            "type": "error",
            "data": {
                "error": error,
                "context": context
            }
        }, critical=True)

    def log_result(self, result: str) -> None:
        """Log final result"""
        self._write_log({
            "type": "result",
            "data": {
                "result": result,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds()
            }
        })

    def save_session(self) -> None:
        """Save session metadata"""
        # Flush logs before saving metadata
        try:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())
        except Exception as e:
            import logging
            logging.warning(f"Failed to flush logs before saving metadata: {e}")

        metadata_file = self.log_dir / f"session_{self.session_id}_metadata.json"
        metadata = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_logs": len(self.logs),
            "log_file": str(self.session_file)
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def close(self) -> None:
        """Explicitly close the log file"""
        self._cleanup()

    def get_session_path(self) -> Path:
        """Get the session log file path"""
        return self.session_file

    def get_context(self, last_n: int = 10) -> list[dict[str, Any]]:
        """Get last N log entries for context"""
        return self.logs[-last_n:] if self.logs else []
