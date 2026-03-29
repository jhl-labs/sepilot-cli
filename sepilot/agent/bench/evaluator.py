import json
from pathlib import Path

from sepilot.agent.bench.models import BenchResult, EvaluationResult


class Evaluator:
    def __init__(self, logger):
        self.logger = logger

    def load_predictions(self, file_path: str) -> list[BenchResult]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {file_path}")

        results = []

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        results.append(BenchResult(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        raise ValueError(f"Failed to parse line: {e}") from e
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data_list = json.load(f)
            if isinstance(data_list, list):
                results = [BenchResult(**item) for item in data_list]
            elif isinstance(data_list, dict):
                results = [BenchResult(**data_list)]
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. Use .json or .jsonl"
            )

        return results

    def evaluate_predictions(
        self, predictions_file: str, dataset_file: str | None = None
    ) -> EvaluationResult:
        results = self.load_predictions(predictions_file)

        passed = sum(1 for r in results if r.test_result == "passed" or r.resolved)
        failed = sum(
            1 for r in results
            if r.test_result == "failed" and not r.resolved
        )
        errors = sum(1 for r in results if r.test_result == "error")

        total = len(results)
        pass_rate = passed / total * 100 if total > 0 else 0.0

        per_instance = {r.instance_id: r.model_dump() for r in results}

        return EvaluationResult(
            total_instances=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=pass_rate,
            per_instance=per_instance,
        )

    def merge_harness_results(
        self,
        results: list[BenchResult],
        harness_results: dict,
    ) -> list[BenchResult]:
        """swebench harness 평가 결과를 BenchResult에 병합."""
        for br in results:
            if br.instance_id in harness_results:
                entry = harness_results[br.instance_id]
                if isinstance(entry, dict):
                    br.resolved = entry.get("resolved", False)
                elif isinstance(entry, bool):
                    br.resolved = entry
                else:
                    continue
                if br.resolved:
                    br.test_result = "passed"
                elif br.model_patch:
                    br.test_result = "failed"
        return results
