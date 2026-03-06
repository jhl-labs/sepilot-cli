"""Error recovery strategies for LangGraph agent execution"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(Enum):
    """Error categories for classification"""
    RECURSION = "recursion"  # GraphRecursionError
    TIMEOUT = "timeout"  # asyncio.TimeoutError
    NETWORK_TIMEOUT = "network_timeout"  # Network-related timeout
    RATE_LIMIT = "rate_limit"  # API rate limit exceeded
    USER_INTERRUPT = "user_interrupt"  # KeyboardInterrupt
    TOOL_ERROR = "tool_error"  # Tool execution failure
    LLM_ERROR = "llm_error"  # LLM API error
    UNKNOWN = "unknown"  # Uncategorized error


@dataclass
class ErrorContext:
    """Context information for an error"""
    error: Exception
    category: ErrorCategory
    attempt_number: int
    total_attempts: int
    error_message: str
    can_retry: bool
    suggested_action: str | None = None
    backoff_seconds: float = 0.0


class ErrorRecoveryStrategy:
    """Handles error categorization and recovery strategies"""

    # Maximum retry attempts by error category
    MAX_RETRIES = {
        ErrorCategory.RATE_LIMIT: 3,
        ErrorCategory.NETWORK_TIMEOUT: 3,
        ErrorCategory.TIMEOUT: 2,
        ErrorCategory.TOOL_ERROR: 1,
        ErrorCategory.LLM_ERROR: 2,
        ErrorCategory.RECURSION: 0,  # Not retryable without intervention
        ErrorCategory.USER_INTERRUPT: 0,  # Not retryable
        ErrorCategory.UNKNOWN: 1,
    }

    # Base backoff times in seconds
    BASE_BACKOFF = {
        ErrorCategory.RATE_LIMIT: 60.0,  # 1 minute for rate limits
        ErrorCategory.NETWORK_TIMEOUT: 5.0,
        ErrorCategory.TIMEOUT: 2.0,
        ErrorCategory.TOOL_ERROR: 1.0,
        ErrorCategory.LLM_ERROR: 3.0,
        ErrorCategory.RECURSION: 0.0,
        ErrorCategory.USER_INTERRUPT: 0.0,
        ErrorCategory.UNKNOWN: 2.0,
    }

    @staticmethod
    def categorize_error(error: BaseException) -> ErrorCategory:
        """Categorize an error for appropriate handling

        Args:
            error: Exception to categorize

        Returns:
            ErrorCategory enum value
        """
        error_str = str(error).lower()

        # Check error type using isinstance where possible, with __name__ fallback
        is_recursion = False
        try:
            from langgraph.errors import GraphRecursionError
            is_recursion = isinstance(error, GraphRecursionError)
        except ImportError:
            pass
        if not is_recursion:
            is_recursion = type(error).__name__ == "GraphRecursionError"
        if is_recursion:
            return ErrorCategory.RECURSION

        if isinstance(error, KeyboardInterrupt):
            return ErrorCategory.USER_INTERRUPT

        error_type = type(error).__name__
        if "timeout" in error_type.lower():
            return ErrorCategory.TIMEOUT

        # Check error message
        if "rate" in error_str and "limit" in error_str:
            return ErrorCategory.RATE_LIMIT
        elif "timeout" in error_str:
            if "network" in error_str or "connection" in error_str:
                return ErrorCategory.NETWORK_TIMEOUT
            else:
                return ErrorCategory.TIMEOUT
        elif any(pat in error_str for pat in ["eof", "500", "connection reset", "connection aborted",
                                               "remote disconnected", "closed connection", "broken pipe",
                                               "sslerror", "ssl:", "certificate_verify_failed",
                                               "max retries exceeded"]):
            return ErrorCategory.NETWORK_TIMEOUT
        elif "tool" in error_str or "execution" in error_str:
            return ErrorCategory.TOOL_ERROR
        elif "api" in error_str or "llm" in error_str or "model" in error_str:
            return ErrorCategory.LLM_ERROR

        return ErrorCategory.UNKNOWN

    @staticmethod
    def can_retry(category: ErrorCategory, attempt_number: int) -> bool:
        """Check if error can be retried

        Args:
            category: Error category
            attempt_number: Current attempt number (1-indexed)

        Returns:
            True if retry is possible
        """
        max_retries = ErrorRecoveryStrategy.MAX_RETRIES.get(category, 0)
        return attempt_number <= max_retries

    @staticmethod
    def get_backoff_time(category: ErrorCategory, attempt_number: int) -> float:
        """Calculate exponential backoff time

        Args:
            category: Error category
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Seconds to wait before retry
        """
        base = ErrorRecoveryStrategy.BASE_BACKOFF.get(category, 2.0)
        # Exponential backoff: base * (2 ^ (attempt - 1))
        return base * (2 ** (attempt_number - 1))

    @staticmethod
    def get_recovery_action(category: ErrorCategory, attempt_number: int) -> str | None:
        """Get suggested recovery action for error

        Args:
            category: Error category
            attempt_number: Current attempt number

        Returns:
            Human-readable recovery suggestion or None
        """
        if category == ErrorCategory.RECURSION:
            return (
                "Task is too complex for current iteration limit. "
                "Consider:\n"
                "  - Breaking task into smaller steps\n"
                "  - Increasing max_iterations setting\n"
                "  - Simplifying the request"
            )
        elif category == ErrorCategory.RATE_LIMIT:
            backoff = ErrorRecoveryStrategy.get_backoff_time(category, attempt_number)
            return f"API rate limit exceeded. Waiting {backoff:.0f}s before retry {attempt_number}"
        elif category == ErrorCategory.NETWORK_TIMEOUT:
            return f"Network timeout. Retrying (attempt {attempt_number})"
        elif category == ErrorCategory.TIMEOUT:
            return f"Operation timeout. Retrying with increased timeout (attempt {attempt_number})"
        elif category == ErrorCategory.TOOL_ERROR:
            return "Tool execution failed. Retrying with error context"
        elif category == ErrorCategory.LLM_ERROR:
            return f"LLM API error. Retrying (attempt {attempt_number})"
        elif category == ErrorCategory.USER_INTERRUPT:
            return "Execution interrupted by user"

        return None

    @staticmethod
    def create_error_context(
        error: BaseException,
        attempt_number: int = 1
    ) -> ErrorContext:
        """Create comprehensive error context

        Args:
            error: The exception that occurred
            attempt_number: Current attempt number

        Returns:
            ErrorContext with all relevant information
        """
        category = ErrorRecoveryStrategy.categorize_error(error)
        can_retry_flag = ErrorRecoveryStrategy.can_retry(category, attempt_number)
        suggested_action = ErrorRecoveryStrategy.get_recovery_action(category, attempt_number)
        backoff = ErrorRecoveryStrategy.get_backoff_time(category, attempt_number) if can_retry_flag else 0.0
        max_retries = ErrorRecoveryStrategy.MAX_RETRIES.get(category, 0)

        return ErrorContext(
            error=error,
            category=category,
            attempt_number=attempt_number,
            total_attempts=max_retries,
            error_message=str(error),
            can_retry=can_retry_flag,
            suggested_action=suggested_action,
            backoff_seconds=backoff
        )

    @staticmethod
    def execute_with_retry(
        operation: Callable[[], Any],
        operation_name: str = "operation",
        on_error: Callable[[ErrorContext], None] | None = None
    ) -> Any:
        """Execute an operation with automatic retry logic

        Args:
            operation: Callable to execute
            operation_name: Name for logging
            on_error: Optional callback for error handling (receives ErrorContext)

        Returns:
            Result from successful operation

        Raises:
            Last exception if all retries exhausted
        """
        attempt = 0
        last_error = None

        while True:
            attempt += 1
            try:
                return operation()

            except Exception as e:
                last_error = e
                context = ErrorRecoveryStrategy.create_error_context(e, attempt)

                # Call error callback if provided
                if on_error:
                    on_error(context)

                # Check if we should retry
                if not context.can_retry:
                    raise

                # Wait before retry (exponential backoff)
                if context.backoff_seconds > 0:
                    time.sleep(context.backoff_seconds)

        # This should never be reached, but satisfy type checker
        raise last_error if last_error else RuntimeError("Unexpected retry loop exit")


# Example usage and testing
if __name__ == "__main__":
    print("=== Error Recovery Strategy Testing ===\n")

    # Test error categorization
    test_errors = [
        (Exception("GraphRecursionError: Max iterations reached"), ErrorCategory.RECURSION),
        (Exception("Rate limit exceeded"), ErrorCategory.RATE_LIMIT),
        (Exception("Network timeout occurred"), ErrorCategory.NETWORK_TIMEOUT),
        (Exception("Tool execution failed"), ErrorCategory.TOOL_ERROR),
        (KeyboardInterrupt(), ErrorCategory.USER_INTERRUPT),
    ]

    print("1. Error Categorization:")
    for error, expected in test_errors:
        category = ErrorRecoveryStrategy.categorize_error(error)
        status = "✓" if category == expected else "✗"
        print(f"  {status} {str(error)[:50]:50} -> {category.value}")

    # Test retry logic
    print("\n2. Retry Logic:")
    for category in [ErrorCategory.RATE_LIMIT, ErrorCategory.RECURSION, ErrorCategory.NETWORK_TIMEOUT]:
        print(f"\n  {category.value}:")
        for attempt in range(1, 5):
            can_retry = ErrorRecoveryStrategy.can_retry(category, attempt)
            backoff = ErrorRecoveryStrategy.get_backoff_time(category, attempt)
            print(f"    Attempt {attempt}: can_retry={can_retry}, backoff={backoff:.1f}s")

    # Test error context creation
    print("\n3. Error Context:")
    test_error = Exception("API rate limit exceeded: 429 Too Many Requests")
    context = ErrorRecoveryStrategy.create_error_context(test_error, attempt_number=2)
    print(f"  Category: {context.category.value}")
    print(f"  Can Retry: {context.can_retry}")
    print(f"  Attempt: {context.attempt_number}/{context.total_attempts}")
    print(f"  Backoff: {context.backoff_seconds:.1f}s")
    print(f"  Suggestion: {context.suggested_action}")

    print("\n✅ All tests completed")
