"""Base tool interface for SE Pilot"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all tools"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def __init__(self, logger=None):
        self.logger = logger

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass

    def get_description(self) -> str:
        """Get formatted description for LLM"""
        param_desc = "\n".join([
            f"  - {name}: {info}"
            for name, info in self.parameters.items()
        ])
        return f"{self.name}: {self.description}\nParameters:\n{param_desc}"

    def validate_params(self, **kwargs) -> bool:
        """Validate that required parameters are provided"""
        required = [k for k, v in self.parameters.items() if "required" in str(v).lower()]
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
        return True
