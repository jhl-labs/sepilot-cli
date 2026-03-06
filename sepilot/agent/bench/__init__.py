from .models import SWEInstance, BenchResult, EvaluationResult, Prediction

__all__ = [
    "SWEInstance",
    "BenchResult",
    "EvaluationResult",
    "Prediction",
    "InferenceRunner",
    "EvaluationRunner",
]


def __getattr__(name):
    """Lazy import for InferenceRunner/EvaluationRunner to avoid docker dependency in containers."""
    if name in ("InferenceRunner", "EvaluationRunner"):
        from .instance_runner import InferenceRunner, EvaluationRunner
        return InferenceRunner if name == "InferenceRunner" else EvaluationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
