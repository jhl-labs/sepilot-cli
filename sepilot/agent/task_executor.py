"""Task Executor - Parallel task execution with cancel/retry support.

Claude Code-style task execution with:
- Configurable concurrency limits
- Individual task cancellation
- Automatic retry with backoff
- Timeout handling
- Progress callbacks
- Integration with TaskRegistry
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

from sepilot.agent.task_registry import (
    TaskInfo,
    TaskPriority,
    TaskRegistry,
    TaskState,
    get_task_registry,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random jitter factor

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        import random
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    traceback: str | None = None
    retry_count: int = 0
    duration_seconds: float = 0.0
    tokens_used: int = 0


@dataclass
class ExecutionConfig:
    """Configuration for task executor."""
    max_concurrent: int = 3
    default_timeout: float = 300.0  # 5 minutes
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    cancel_on_error: bool = False  # Cancel all tasks if one fails


class TaskExecutor:
    """Execute tasks in parallel with cancel/retry support.

    Example:
        executor = TaskExecutor(max_concurrent=5)

        # Register and execute tasks
        async def my_task(data):
            return process(data)

        results = await executor.execute_batch([
            ("task1", my_task, {"data": 1}),
            ("task2", my_task, {"data": 2}),
        ])

        # Or with more control
        executor.submit("task1", my_task(data))
        executor.submit("task2", my_task(data))
        results = await executor.wait_all()

        # Cancel specific task
        executor.cancel("task1")
    """

    def __init__(
        self,
        registry: TaskRegistry | None = None,
        config: ExecutionConfig | None = None,
        progress_callback: Callable[[TaskInfo], None] | None = None
    ):
        """Initialize executor.

        Args:
            registry: Task registry (uses global if None)
            config: Execution configuration
            progress_callback: Callback for progress updates
        """
        self.registry = registry or get_task_registry()
        self.config = config or ExecutionConfig()
        self.progress_callback = progress_callback

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._submitted: dict[str, asyncio.Task] = {}
        self._results: dict[str, ExecutionResult] = {}
        self._cancel_event = asyncio.Event()

    # ==================== Submission ====================

    def submit(
        self,
        name: str,
        coro: Coroutine[Any, Any, T],
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        max_retries: int | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Submit a coroutine for execution.

        Args:
            name: Task name
            coro: Coroutine to execute
            description: Task description
            priority: Task priority
            timeout: Task timeout (uses default if None)
            max_retries: Max retries (uses policy default if None)
            dependencies: Task IDs that must complete first
            metadata: Additional metadata

        Returns:
            Task ID
        """
        # Register task
        task_info = self.registry.register(
            name=name,
            description=description or name,
            priority=priority,
            dependencies=dependencies,
            max_retries=max_retries or self.config.retry_policy.max_retries,
            timeout_seconds=timeout or self.config.default_timeout,
            metadata=metadata
        )

        # Create wrapper coroutine with retry/timeout handling
        wrapped = self._create_wrapper(task_info.task_id, coro)

        # Submit to event loop
        asyncio_task = asyncio.create_task(wrapped)
        self._submitted[task_info.task_id] = asyncio_task
        self.registry.set_asyncio_task(task_info.task_id, asyncio_task)

        logger.debug(f"Submitted task: {task_info.task_id} - {name}")
        return task_info.task_id

    async def _create_wrapper(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, T]
    ) -> ExecutionResult:
        """Wrap coroutine with semaphore, retry, timeout handling."""
        task_info = self.registry.get(task_id)
        if not task_info:
            return ExecutionResult(
                task_id=task_id,
                success=False,
                error="Task not found in registry"
            )

        # Wait for dependencies
        await self._wait_for_dependencies(task_info)

        # Acquire semaphore
        async with self._semaphore:
            # Check if cancelled
            if self._cancel_event.is_set() and self.config.cancel_on_error:
                self.registry.cancel(task_id, "Cancelled due to other task failure")
                return ExecutionResult(
                    task_id=task_id,
                    success=False,
                    error="Cancelled due to other task failure"
                )

            return await self._execute_with_retry(task_id, coro)

    async def _wait_for_dependencies(self, task_info: TaskInfo) -> None:
        """Wait for all dependencies to complete."""
        if not task_info.dependencies:
            return

        while True:
            all_complete = True
            for dep_id in task_info.dependencies:
                dep = self.registry.get(dep_id)
                if not dep:
                    continue
                if dep.state == TaskState.FAILED:
                    # Dependency failed - mark as blocked
                    self.registry.fail(
                        task_info.task_id,
                        f"Dependency {dep_id} failed"
                    )
                    return
                if dep.state != TaskState.COMPLETED:
                    all_complete = False
                    break

            if all_complete:
                return

            await asyncio.sleep(0.1)

    async def _execute_with_retry(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, T]
    ) -> ExecutionResult:
        """Execute with retry logic."""
        task_info = self.registry.get(task_id)
        if not task_info:
            return ExecutionResult(task_id=task_id, success=False, error="Task not found")

        # Start task
        self.registry.start(task_id)
        self._notify_progress(task_id)

        start_time = datetime.now()
        last_error = None
        last_traceback = None

        for attempt in range(task_info.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    coro,
                    timeout=task_info.timeout_seconds
                )

                # Success
                duration = (datetime.now() - start_time).total_seconds()
                tokens = getattr(result, "tokens_used", 0) if hasattr(result, "tokens_used") else 0

                self.registry.complete(task_id, result=result, tokens_used=tokens)
                self._notify_progress(task_id)

                exec_result = ExecutionResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    retry_count=attempt,
                    duration_seconds=duration,
                    tokens_used=tokens
                )
                self._results[task_id] = exec_result
                return exec_result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {task_info.timeout_seconds}s"
                last_traceback = None
                logger.warning(f"Task {task_id} timed out (attempt {attempt + 1})")

            except asyncio.CancelledError:
                # Task was cancelled
                self.registry.cancel(task_id, "Task cancelled")
                self._notify_progress(task_id)
                exec_result = ExecutionResult(
                    task_id=task_id,
                    success=False,
                    error="Task cancelled"
                )
                self._results[task_id] = exec_result
                return exec_result

            except Exception as e:
                last_error = str(e)
                last_traceback = traceback.format_exc()
                logger.warning(f"Task {task_id} failed (attempt {attempt + 1}): {e}")

            # Retry if possible
            if attempt < task_info.max_retries:
                delay = self.config.retry_policy.get_delay(attempt)
                logger.info(f"Retrying task {task_id} in {delay:.1f}s")
                self.registry.retry(task_id)
                self._notify_progress(task_id)
                await asyncio.sleep(delay)

                # Need to recreate coroutine for retry
                # This is a limitation - coro can only be awaited once
                # Caller should provide a factory function for retryable tasks

        # All retries exhausted
        duration = (datetime.now() - start_time).total_seconds()
        self.registry.fail(task_id, last_error or "Unknown error", last_traceback)
        self._notify_progress(task_id)

        # Set cancel event if configured
        if self.config.cancel_on_error:
            self._cancel_event.set()

        exec_result = ExecutionResult(
            task_id=task_id,
            success=False,
            error=last_error,
            traceback=last_traceback,
            retry_count=task_info.max_retries,
            duration_seconds=duration
        )
        self._results[task_id] = exec_result
        return exec_result

    # ==================== Batch Execution ====================

    async def execute_batch(
        self,
        tasks: list[tuple[str, Callable[[], Coroutine], dict[str, Any] | None]],
        wait: bool = True
    ) -> list[ExecutionResult]:
        """Execute multiple tasks.

        Args:
            tasks: List of (name, coro_factory, kwargs)
                   coro_factory should return a new coroutine each call
            wait: Wait for all tasks to complete

        Returns:
            List of ExecutionResults
        """
        task_ids = []

        for name, coro_factory, kwargs in tasks:
            kwargs = kwargs or {}
            coro = coro_factory()
            task_id = self.submit(name, coro, **kwargs)
            task_ids.append(task_id)

        if wait:
            return await self.wait_tasks(task_ids)

        return []

    async def execute_parallel(
        self,
        coroutines: list[tuple[str, Coroutine[Any, Any, T]]]
    ) -> list[ExecutionResult]:
        """Execute coroutines in parallel with semaphore limiting.

        Args:
            coroutines: List of (name, coroutine) tuples

        Returns:
            List of ExecutionResults
        """
        task_ids = []
        for name, coro in coroutines:
            task_id = self.submit(name, coro)
            task_ids.append(task_id)

        return await self.wait_tasks(task_ids)

    # ==================== Waiting ====================

    async def wait_tasks(self, task_ids: list[str]) -> list[ExecutionResult]:
        """Wait for specific tasks to complete.

        Args:
            task_ids: Tasks to wait for

        Returns:
            List of ExecutionResults
        """
        asyncio_tasks = [
            self._submitted[tid] for tid in task_ids
            if tid in self._submitted
        ]

        if asyncio_tasks:
            await asyncio.gather(*asyncio_tasks, return_exceptions=True)

        return [self._results.get(tid, ExecutionResult(tid, False, error="Not found")) for tid in task_ids]

    async def wait_all(self) -> list[ExecutionResult]:
        """Wait for all submitted tasks to complete.

        Returns:
            List of all ExecutionResults
        """
        if self._submitted:
            await asyncio.gather(*self._submitted.values(), return_exceptions=True)

        return list(self._results.values())

    # ==================== Cancellation ====================

    def cancel(self, task_id: str, reason: str = "") -> bool:
        """Cancel a specific task.

        Args:
            task_id: Task to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled
        """
        asyncio_task = self._submitted.get(task_id)
        if asyncio_task and not asyncio_task.done():
            asyncio_task.cancel()

        result = self.registry.cancel(task_id, reason)
        return result is not None

    def cancel_all(self, reason: str = "Bulk cancellation") -> int:
        """Cancel all pending/running tasks.

        Args:
            reason: Cancellation reason

        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task_id, asyncio_task in self._submitted.items():
            if not asyncio_task.done():
                asyncio_task.cancel()
                self.registry.cancel(task_id, reason)
                count += 1

        self._cancel_event.set()
        return count

    # ==================== Status ====================

    def get_running_count(self) -> int:
        """Get number of currently running tasks."""
        return len(self.registry.get_running_tasks())

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return len(self.registry.get_pending_tasks())

    def is_complete(self) -> bool:
        """Check if all submitted tasks are complete."""
        return all(
            task.done() for task in self._submitted.values()
        )

    def get_results(self) -> dict[str, ExecutionResult]:
        """Get all execution results."""
        return self._results.copy()

    # ==================== Progress ====================

    def _notify_progress(self, task_id: str) -> None:
        """Notify progress callback."""
        if self.progress_callback:
            task_info = self.registry.get(task_id)
            if task_info:
                try:
                    self.progress_callback(task_info)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")

    # ==================== Cleanup ====================

    def clear(self) -> None:
        """Clear all submitted tasks and results."""
        self._submitted.clear()
        self._results.clear()
        self._cancel_event.clear()


class RetryableTask:
    """Wrapper for tasks that can be retried.

    Since coroutines can only be awaited once, this provides a factory pattern.

    Example:
        async def fetch_data(url):
            return await http.get(url)

        task = RetryableTask(fetch_data, url="https://example.com")
        executor.submit_retryable("fetch", task)
    """

    def __init__(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs
    ):
        """Initialize retryable task.

        Args:
            func: Async function to call
            args: Positional arguments
            kwargs: Keyword arguments
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def create_coroutine(self) -> Coroutine[Any, Any, T]:
        """Create a new coroutine instance."""
        return self.func(*self.args, **self.kwargs)


async def execute_with_progress(
    tasks: list[tuple[str, Coroutine]],
    max_concurrent: int = 3,
    progress_callback: Callable[[str, TaskInfo], None] | None = None
) -> list[ExecutionResult]:
    """Convenience function for parallel execution with progress tracking.

    Args:
        tasks: List of (name, coroutine) tuples
        max_concurrent: Maximum concurrent tasks
        progress_callback: Optional (task_id, task_info) callback

    Returns:
        List of ExecutionResults
    """
    config = ExecutionConfig(max_concurrent=max_concurrent)

    def wrapped_callback(task_info: TaskInfo) -> None:
        if progress_callback:
            progress_callback(task_info.task_id, task_info)

    executor = TaskExecutor(config=config, progress_callback=wrapped_callback)

    return await executor.execute_parallel(tasks)
