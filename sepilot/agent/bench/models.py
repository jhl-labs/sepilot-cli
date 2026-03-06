import json

from pydantic import BaseModel, Field, field_validator
from typing import Any


class SWEInstance(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    patch: str = ""
    test_patch: str = ""
    problem_statement: str
    hints_text: str = ""
    created_at: str = ""
    version: str = ""
    FAIL_TO_PASS: list[str] = Field(default_factory=list)
    PASS_TO_PASS: list[str] = Field(default_factory=list)
    environment_setup_commit: str = ""

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS", mode="before")
    @classmethod
    def parse_json_string(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class Prediction(BaseModel):
    """SWE-bench 호환 예측 형식 (predictions.jsonl)"""
    instance_id: str
    model_name_or_path: str = "sepilot"
    model_patch: str = ""


class BenchResult(BaseModel):
    instance_id: str
    status: str = "pending"
    model_patch: str | None = None
    test_result: str = "skipped"
    duration_seconds: float = 0.0
    logs: str = ""
    error_message: str | None = None
    resolved: bool = False


class EvaluationResult(BaseModel):
    total_instances: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    per_instance: dict[str, dict[str, Any]]
