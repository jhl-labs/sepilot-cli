"""SWE-bench 벤치마크 에이전트.

2단계 워크플로우 오케스트레이션:
  Phase 1 (Inference): Docker 컨테이너에서 에이전트 실행 -> 패치 생성
  Phase 2 (Evaluation): swebench harness로 패치 평가
"""

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from sepilot.agent.bench.dataset_loader import DatasetLoader
from sepilot.agent.bench.datasets import get_dataset_config, PREDEFINED_DATASETS, get_preset_config
from sepilot.agent.bench.evaluator import Evaluator
from sepilot.agent.bench.instance_runner import InferenceRunner, EvaluationRunner
from sepilot.agent.bench.models import SWEInstance, BenchResult


class BenchAgent:
    def __init__(self, settings, logger, console: Console):
        self.settings = settings
        self.logger = logger
        self.console = console

        self.instances: list[SWEInstance] = []
        self.dataset_source: str = ""
        self.last_run_results: list[BenchResult] = []
        self.last_run_timestamp: str = ""
        self.last_predictions_path: Path | None = None
        self.last_evaluation_results: dict | None = None

        self.inference_runner = InferenceRunner(logger, console)
        self.evaluation_runner = EvaluationRunner(logger, console)
        self.evaluator = Evaluator(logger)
        self._team_mode: bool = False

    def load_instances_from_config(self, dataset_config: dict):
        """config dict에서 직접 인스턴스 로드."""
        self.instances = DatasetLoader.load_predefined_dataset(dataset_config)
        self.dataset_source = f"predefined:{dataset_config['name']}"
        self.console.print(
            f"[green]Loaded preset: {dataset_config['name']} "
            f"({len(self.instances)} instances from "
            f"{dataset_config.get('dataset_name', 'SWE-bench')})[/green]"
        )

    def load_instances(self, data_source: str):
        dataset_config = get_dataset_config(data_source)

        if dataset_config:
            self.instances = DatasetLoader.load_predefined_dataset(dataset_config)
            self.dataset_source = f"predefined:{data_source}"
            self.console.print(
                f"[green]Loaded dataset: {data_source} ({len(self.instances)} instances)[/green]"
            )
        else:
            self.instances = DatasetLoader.load_from_file(data_source)
            self.dataset_source = f"file:{data_source}"
            self.console.print(
                f"[green]Loaded {len(self.instances)} instances from: {data_source}[/green]"
            )

    def enrich_instances(self, full_dataset_path: str):
        pass

    def ensure_images(self):
        """Docker 이미지 사전 빌드."""
        if not self.instances:
            self.console.print(
                "[yellow]No instances loaded. Use /bench instances load first.[/yellow]"
            )
            return

        self.inference_runner.ensure_images(self.instances)

    def run_instances(
        self,
        limit: int | None = None,
        run_tests: bool = False,
        max_workers: int = 1,
        timeout: int = 600,
        team_mode: bool = False,
    ):
        """Phase 1: 에이전트 실행하여 패치 생성."""
        if not self.instances:
            self.console.print(
                "[yellow]No instances loaded. Use /bench instances load first.[/yellow]"
            )
            return

        instances_to_run = self.instances[:limit] if limit else self.instances
        mode_label = "팀 모드" if team_mode else "단일 에이전트"
        self.console.print(
            f"[cyan]Phase 1: Running inference for {len(instances_to_run)} instances "
            f"({mode_label})...[/cyan]"
        )

        self.last_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # InferenceRunner에 team_mode 반영 (매번 새로 생성하여 상태 반영)
        self.inference_runner = InferenceRunner(self.logger, self.console, team_mode=team_mode)

        # Docker 이미지 확인/빌드
        self.inference_runner.ensure_images(instances_to_run)

        # 에이전트 실행
        self.last_run_results = self.inference_runner.run_batch(
            instances_to_run,
            max_workers=max_workers,
            timeout=timeout,
        )

        # predictions.jsonl 저장
        self.last_predictions_path = self.inference_runner.save_predictions(
            self.last_run_results
        )

        self._print_summary()

    def evaluate_predictions(
        self,
        predictions_file: str | None = None,
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "test",
        max_workers: int = 4,
        timeout: int = 900,
    ) -> dict:
        """Phase 2: swebench harness로 평가."""
        if predictions_file:
            pred_path = Path(predictions_file)
        elif self.last_predictions_path:
            pred_path = self.last_predictions_path
        else:
            # 최신 predictions 파일 찾기
            latest = self.inference_runner.PREDICTIONS_DIR / "predictions_latest.jsonl"
            if latest.exists():
                pred_path = latest.resolve()
            else:
                self.console.print(
                    "[yellow]No predictions file found. Run /bench run first.[/yellow]"
                )
                return {}

        self.console.print(
            f"[cyan]Phase 2: Evaluating predictions from {pred_path}...[/cyan]"
        )

        try:
            results = self.evaluation_runner.evaluate(
                predictions_path=pred_path,
                dataset_name=dataset_name,
                split=split,
                max_workers=max_workers,
                timeout=timeout,
            )

            self.last_evaluation_results = results

            # BenchResult에 resolved 상태 업데이트
            for br in self.last_run_results:
                if br.instance_id in results:
                    br.resolved = results[br.instance_id].get("resolved", False)
                    if br.resolved:
                        br.test_result = "passed"
                    else:
                        br.test_result = "failed"

            return results

        except Exception as e:
            self.console.print(f"[red]Evaluation failed: {e}[/red]")
            raise

    def export_jsonl(self, source: str | None, output: str):
        if not self.last_run_results:
            self.console.print(
                "[yellow]No run results to export. Run /bench run first.[/yellow]"
            )
            return

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.last_run_results:
                data = {
                    "instance_id": result.instance_id,
                    "model_name_or_path": "sepilot",
                    "model_patch": result.model_patch or "",
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        self.console.print(
            f"[green]Exported {len(self.last_run_results)} results to: {output_path}[/green]"
        )

    def get_status(self) -> dict:
        dataset_name = "unknown"
        if "test7" in self.dataset_source:
            dataset_name = "test7"
        elif "test30" in self.dataset_source:
            dataset_name = "test30"
        elif "test50" in self.dataset_source:
            dataset_name = "test50"
        elif "file:" in self.dataset_source:
            dataset_name = self.dataset_source.replace("file:", "")

        status = {
            "dataset": dataset_name,
            "count": len(self.instances),
            "source": self.dataset_source,
            "last_run": len(self.last_run_results),
            "last_run_timestamp": self.last_run_timestamp,
        }

        if self.last_predictions_path:
            status["predictions_file"] = str(self.last_predictions_path)

        if self.last_evaluation_results:
            resolved = sum(
                1 for r in self.last_evaluation_results.values()
                if r.get("resolved")
            )
            status["evaluation"] = f"{resolved}/{len(self.last_evaluation_results)} resolved"

        return status

    def _print_summary(self):
        if not self.last_run_results:
            return

        total = len(self.last_run_results)
        with_patch = sum(1 for r in self.last_run_results if r.model_patch)
        errors = sum(1 for r in self.last_run_results if r.status == "error")
        total_time = sum(r.duration_seconds for r in self.last_run_results)

        self.console.print(f"\n[bold]Inference Summary:[/bold]")
        self.console.print(f"  Total instances: {total}")
        self.console.print(f"  [green]With patch: {with_patch}[/green]")
        self.console.print(f"  [yellow]No patch: {total - with_patch - errors}[/yellow]")
        self.console.print(f"  [red]Errors: {errors}[/red]")
        self.console.print(f"  Total time: {total_time:.2f}s")
        if total > 0:
            self.console.print(f"  Avg time: {total_time / total:.2f}s per instance")
        if self.last_predictions_path:
            self.console.print(
                f"  [dim]Predictions: {self.last_predictions_path}[/dim]"
            )
