"""Multi-agent orchestration data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MessageType(str, Enum):
    """Type of inter-agent message."""

    TASK = "task"
    RESULT = "result"
    FEEDBACK = "feedback"
    COORDINATE = "coordinate"


class PMAction(str, Enum):
    """PM decision action types."""

    DONE = "done"
    RETRY = "retry"
    COORDINATE = "coordinate"
    ABORT = "abort"


class Strategy(str, Enum):
    """Team execution strategy."""

    AUTO = "auto"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PIPELINE = "pipeline"


@dataclass
class Message:
    """A message exchanged between agents."""

    sender: str
    receiver: str
    content: str
    msg_type: MessageType
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    reply_to: str | None = None


@dataclass
class AgentRole:
    """Definition of an agent role in a team."""

    name: str
    agent_cmd: str
    system_prompt: str = ""
    available: bool = True
    workdir: str | None = None


@dataclass
class TeamPreset:
    """A preset team configuration."""

    name: str
    description: str
    roles: list[AgentRole] = field(default_factory=list)
    strategy: str = Strategy.AUTO


@dataclass
class TaskAssignment:
    """A task assigned to a specific role."""

    role_name: str
    task_description: str
    depends_on: list[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class PMDecision:
    """A decision made by the PM agent after reviewing results."""

    action: str  # PMAction value
    reason: str
    retry_targets: list[str] = field(default_factory=list)
    retry_instructions: dict[str, str] = field(default_factory=dict)
    coordinate_pairs: list[tuple[str, str]] = field(default_factory=list)
