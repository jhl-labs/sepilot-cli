"""State update builder pattern for cleaner state management"""

from datetime import datetime
from typing import Any


class StateUpdateBuilder:
    """Builder pattern for constructing state updates

    This class provides a fluent interface for building state update dictionaries,
    reducing code duplication and making state updates more maintainable.

    Example:
        >>> builder = StateUpdateBuilder()
        >>> updates = (builder
        ...     .set_decision("graph", "Complex task requires planning")
        ...     .set_plan(["Step 1", "Step 2"], ["Note about complexity"])
        ...     .track_tokens(150, 0.003)
        ...     .build())
    """

    def __init__(self):
        """Initialize empty state update builder"""
        self.updates: dict[str, Any] = {}

    def set_decision(self, decision: str, reason: str | None = None) -> "StateUpdateBuilder":
        """Set triage decision

        Args:
            decision: Decision type (e.g., 'graph', 'direct')
            reason: Optional reason for decision

        Returns:
            Self for method chaining
        """
        self.updates["triage_decision"] = decision
        if reason:
            self.updates["triage_reason"] = reason
        return self

    def set_plan(
        self,
        steps: list[str],
        notes: list[str] | None = None,
        complexity: float | None = None
    ) -> "StateUpdateBuilder":
        """Set planning information

        Args:
            steps: List of plan steps
            notes: Optional planning notes
            complexity: Optional complexity estimate (0.0-1.0)

        Returns:
            Self for method chaining
        """
        self.updates["plan_created"] = True
        self.updates["plan_steps"] = steps
        if notes:
            self.updates["planning_notes"] = notes
        if complexity is not None:
            self.updates["plan_complexity"] = complexity
        return self

    def track_tokens(self, tokens: int, cost: float = 0.0) -> "StateUpdateBuilder":
        """Track token usage and cost

        Args:
            tokens: Number of tokens used
            cost: Estimated cost in USD

        Returns:
            Self for method chaining
        """
        # These will be added to existing values via state reducers
        self.updates["total_tokens_used"] = tokens
        if cost > 0:
            self.updates["estimated_cost"] = cost
        return self

    def add_message(self, message: Any) -> "StateUpdateBuilder":
        """Add a message to state

        Args:
            message: Message object (BaseMessage or compatible)

        Returns:
            Self for method chaining
        """
        if "messages" not in self.updates:
            self.updates["messages"] = []
        self.updates["messages"].append(message)
        return self

    def add_messages(self, messages: list[Any]) -> "StateUpdateBuilder":
        """Add multiple messages to state

        Args:
            messages: List of message objects

        Returns:
            Self for method chaining
        """
        if "messages" not in self.updates:
            self.updates["messages"] = []
        self.updates["messages"].extend(messages)
        return self

    def set_node_status(
        self,
        node_name: str,
        status: str = "completed",
        error: str | None = None
    ) -> "StateUpdateBuilder":
        """Set status for a specific node

        Args:
            node_name: Name of the node
            status: Status string (e.g., 'completed', 'error', 'executing')
            error: Optional error message

        Returns:
            Self for method chaining
        """
        if "node_status" not in self.updates:
            self.updates["node_status"] = {}
        self.updates["node_status"][node_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        return self

    def increment_iteration(self, delta: int = 1) -> "StateUpdateBuilder":
        """Increment iteration counter

        Args:
            delta: Amount to increment by (default: 1)

        Returns:
            Self for method chaining
        """
        # Note: This assumes the state has a reducer that adds to existing value
        self.updates["iteration_count"] = delta
        return self

    def set_execution_pending(self, pending: bool = True) -> "StateUpdateBuilder":
        """Set plan execution pending flag

        Args:
            pending: Whether execution is pending

        Returns:
            Self for method chaining
        """
        self.updates["plan_execution_pending"] = pending
        return self

    def track_file_changes(
        self,
        files_added: list[str],
        files_modified: list[str],
        files_deleted: list[str] | None = None
    ) -> "StateUpdateBuilder":
        """Track file system changes

        Args:
            files_added: List of added file paths
            files_modified: List of modified file paths
            files_deleted: Optional list of deleted file paths

        Returns:
            Self for method chaining
        """
        total_changes = len(files_added) + len(files_modified)
        if files_deleted:
            total_changes += len(files_deleted)

        self.updates["file_changes_count"] = total_changes

        # Track specific files
        if "modified_files" not in self.updates:
            self.updates["modified_files"] = []
        self.updates["modified_files"].extend(files_added + files_modified)

        return self

    def set_force_termination(self, reason: str | None = None) -> "StateUpdateBuilder":
        """Set force termination flag

        Args:
            reason: Optional reason for termination

        Returns:
            Self for method chaining
        """
        self.updates["force_termination"] = True
        if reason:
            self.updates["termination_reason"] = reason
        return self

    def set_verification_result(
        self,
        passed: bool,
        message: str | None = None,
        confidence: float | None = None
    ) -> "StateUpdateBuilder":
        """Set verification results

        Args:
            passed: Whether verification passed
            message: Optional verification message
            confidence: Optional confidence score (0.0-1.0)

        Returns:
            Self for method chaining
        """
        self.updates["verification_passed"] = passed
        if message:
            self.updates["verification_message"] = message
        if confidence is not None:
            self.updates["verification_confidence"] = confidence
        return self

    def merge(self, other_updates: dict[str, Any]) -> "StateUpdateBuilder":
        """Merge another update dictionary (deep merge for lists and dicts)

        Args:
            other_updates: Dictionary to merge

        Returns:
            Self for method chaining
        """
        for key, value in other_updates.items():
            if key not in self.updates:
                self.updates[key] = value
            elif isinstance(self.updates[key], list) and isinstance(value, list):
                self.updates[key] = self.updates[key] + value
            elif isinstance(self.updates[key], dict) and isinstance(value, dict):
                merged = self.updates[key].copy()
                merged.update(value)
                self.updates[key] = merged
            else:
                self.updates[key] = value
        return self

    def build(self) -> dict[str, Any]:
        """Build and return the final state update dictionary

        Returns:
            Dictionary of state updates (copy of internal dict)
        """
        return dict(self.updates)

    def reset(self) -> "StateUpdateBuilder":
        """Reset the builder to empty state

        Returns:
            Self for method chaining
        """
        self.updates = {}
        return self


# Convenience function for quick single-use builders
def create_state_update(**kwargs) -> dict[str, Any]:
    """Create a state update dictionary from keyword arguments

    This is a convenience function for simple cases where the builder
    pattern would be overkill.

    Args:
        **kwargs: Key-value pairs for state update

    Returns:
        Dictionary suitable for state update

    Example:
        >>> updates = create_state_update(
        ...     triage_decision="graph",
        ...     plan_created=True
        ... )
    """
    return kwargs


# Example usage and testing
if __name__ == "__main__":
    print("=== State Update Builder Testing ===\n")

    # Test 1: Basic builder usage
    print("1. Basic Builder Usage:")
    builder = StateUpdateBuilder()
    updates = (builder
        .set_decision("graph", "Complex task requires planning")
        .set_plan(
            steps=["Analyze requirements", "Implement solution", "Test"],
            notes=["High complexity task"],
            complexity=0.8
        )
        .track_tokens(150, 0.003)
        .build())

    print(f"  Decision: {updates.get('triage_decision')}")
    print(f"  Plan created: {updates.get('plan_created')}")
    print(f"  Steps: {len(updates.get('plan_steps', []))}")
    print(f"  Tokens: {updates.get('total_tokens_used')}")
    print(f"  Cost: ${updates.get('estimated_cost', 0):.4f}")

    # Test 2: Method chaining
    print("\n2. Method Chaining:")
    updates2 = (StateUpdateBuilder()
        .set_node_status("planner", "completed")
        .increment_iteration(1)
        .set_execution_pending(True)
        .build())

    print(f"  Node status keys: {list(updates2.get('node_status', {}).keys())}")
    print(f"  Iteration increment: {updates2.get('iteration_count')}")
    print(f"  Execution pending: {updates2.get('plan_execution_pending')}")

    # Test 3: File change tracking
    print("\n3. File Change Tracking:")
    updates3 = (StateUpdateBuilder()
        .track_file_changes(
            files_added=["new_file.py"],
            files_modified=["existing.py", "config.yaml"]
        )
        .build())

    print(f"  File changes count: {updates3.get('file_changes_count')}")
    print(f"  Modified files: {updates3.get('modified_files')}")

    # Test 4: Verification results
    print("\n4. Verification Results:")
    updates4 = (StateUpdateBuilder()
        .set_verification_result(
            passed=True,
            message="All tests passed",
            confidence=0.95
        )
        .build())

    print(f"  Passed: {updates4.get('verification_passed')}")
    print(f"  Message: {updates4.get('verification_message')}")
    print(f"  Confidence: {updates4.get('verification_confidence')}")

    # Test 5: Builder reset and reuse
    print("\n5. Builder Reset and Reuse:")
    builder = StateUpdateBuilder()
    updates5a = builder.set_decision("direct").build()
    print(f"  First build - decision: {updates5a.get('triage_decision')}")

    builder.reset()
    updates5b = builder.set_decision("graph").build()
    print(f"  After reset - decision: {updates5b.get('triage_decision')}")
    print(f"  Previous updates unchanged: {updates5a.get('triage_decision')}")

    # Test 6: Merge functionality
    print("\n6. Merge Functionality:")
    base_updates = {"existing_key": "existing_value"}
    updates6 = (StateUpdateBuilder()
        .merge(base_updates)
        .set_decision("graph")
        .build())

    print(f"  Merged keys: {list(updates6.keys())}")
    print(f"  Existing value preserved: {updates6.get('existing_key')}")

    print("\n✅ All tests completed")
