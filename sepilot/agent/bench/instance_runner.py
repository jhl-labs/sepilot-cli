"""SWE-bench 인스턴스 실행 및 평가 모듈.

Phase 1 (InferenceRunner): SWE-bench Docker 컨테이너에서 에이전트 실행 -> 패치 생성
Phase 2 (EvaluationRunner): swebench.harness.run_evaluation으로 패치 평가
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from rich.console import Console

from sepilot.agent.bench.models import SWEInstance, BenchResult, Prediction


class InferenceRunner:
    """Phase 1: SWE-bench Docker 컨테이너에서 에이전트를 실행하여 패치 생성."""

    BASE_WORK_DIR = Path(tempfile.gettempdir()) / "sepilot-bench"
    OUTPUT_DIR = BASE_WORK_DIR / "output"
    PREDICTIONS_DIR = BASE_WORK_DIR / "predictions"
    CONTAINER_AGENT_VENV = "/testbed/sepilot_agent_venv"
    CONTAINER_PROBLEM_FILE = "/testbed/sepilot_problem.txt"
    CONTAINER_RESULT_FILE = "/testbed/sepilot_result.json"
    CONTAINER_OUTPUT_DIR = "/testbed/sepilot_output"
    CONTAINER_PATCH_FILE = "/testbed/sepilot_patch.diff"

    def __init__(self, logger, console: Console, team_mode: bool = False):
        self.logger = logger
        self.console = console
        self.team_mode = team_mode
        # 프로젝트 루트: sepilot-cli 디렉토리
        self.project_dir = Path(__file__).resolve().parent.parent.parent.parent

        self.BASE_WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def ensure_images(self, instances: list[SWEInstance]) -> dict[str, str]:
        """swebench.harness.prepare_images로 Docker 이미지 빌드.

        Returns:
            instance_id -> image_key 매핑
        """
        instance_ids = [inst.instance_id for inst in instances]
        self.console.print(
            f"[cyan]Preparing Docker images for {len(instance_ids)} instances...[/cyan]"
        )

        try:
            from swebench.harness.prepare_images import main as prepare_images

            prepare_images(
                dataset_name="princeton-nlp/SWE-bench",
                split="test",
                instance_ids=instance_ids,
                max_workers=4,
                force_rebuild=False,
                open_file_limit=4096,
                namespace=None,
                tag="latest",
                env_image_tag="latest",
            )
        except Exception as e:
            self.console.print(f"[yellow]Image preparation warning: {e}[/yellow]")
            self.console.print("[dim]Attempting to use existing images...[/dim]")

        # TestSpec으로 이미지 키 조회
        image_map = {}
        for inst in instances:
            image_key = self._get_instance_image_key(inst)
            image_map[inst.instance_id] = image_key

        # 이미지 존재 확인
        try:
            client = docker.from_env()
            for instance_id, image_key in image_map.items():
                try:
                    client.images.get(image_key)
                    self.console.print(f"  [green]{instance_id}[/green]: {image_key}")
                except docker.errors.ImageNotFound:
                    self.console.print(
                        f"  [red]{instance_id}[/red]: {image_key} (not found)"
                    )
        except Exception as e:
            self.console.print(f"[yellow]Docker check warning: {e}[/yellow]")

        return image_map

    def _get_instance_image_key(self, instance: SWEInstance) -> str:
        """TestSpec을 통해 SWE-bench 인스턴스 이미지 키를 조회."""
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec

            swebench_instance = self._to_swebench_instance(instance)
            test_spec = make_test_spec(swebench_instance)
            return test_spec.instance_image_key
        except Exception:
            # 폴백: 표준 명명 규칙 사용
            return f"sweb.eval.x86_64.{instance.instance_id}:latest"

    def _to_swebench_instance(self, instance: SWEInstance) -> dict:
        """SWEInstance를 SWEbenchInstance TypedDict 호환 dict로 변환.

        FAIL_TO_PASS/PASS_TO_PASS를 JSON 문자열로 변환.
        """
        data = instance.model_dump()
        # swebench는 이 필드들을 JSON 문자열로 기대
        if isinstance(data.get("FAIL_TO_PASS"), list):
            data["FAIL_TO_PASS"] = json.dumps(data["FAIL_TO_PASS"])
        if isinstance(data.get("PASS_TO_PASS"), list):
            data["PASS_TO_PASS"] = json.dumps(data["PASS_TO_PASS"])
        return data

    def run_instance(
        self,
        instance: SWEInstance,
        timeout: int = 600,
    ) -> BenchResult:
        """단일 인스턴스: Docker 컨테이너에서 에이전트 실행."""
        start_time = time.time()
        logs = []
        instance_id = instance.instance_id

        logs.append(f"Processing: {instance_id}")
        logs.append(f"Repo: {instance.repo}")
        logs.append(f"Base commit: {instance.base_commit}")

        try:
            client = docker.from_env()
            image_key = self._get_instance_image_key(instance)

            # 이미지 존재 확인
            try:
                client.images.get(image_key)
            except docker.errors.ImageNotFound:
                error_msg = f"Docker image not found: {image_key}"
                logs.append(f"Error: {error_msg}")
                self.console.print(f"[red]{instance_id}: {error_msg}[/red]")
                return BenchResult(
                    instance_id=instance_id,
                    status="error",
                    error_message=error_msg,
                    duration_seconds=time.time() - start_time,
                    logs="\n".join(logs),
                )

            # 문제 설명 파일 생성 (PID 포함 → 병렬 실행 시 충돌 방지)
            problem_file = self.BASE_WORK_DIR / f"problem_{instance_id}_{os.getpid()}.txt"
            problem_file.write_text(instance.problem_statement, encoding="utf-8")
            # Docker bind mount는 호스트 파일이 없으면 디렉토리로 마운트 → IsADirectoryError
            if not problem_file.is_file():
                raise RuntimeError(f"Problem file not created: {problem_file}")

            result_file = self.OUTPUT_DIR / f"result_{instance_id}.json"
            if result_file.exists():
                result_file.unlink()

            # 환경변수 설정
            env_vars = self._build_env_vars()

            # Docker 컨테이너 실행
            self.console.print(
                f"[dim]Running agent in container for {instance_id} "
                f"(image: {image_key})...[/dim]"
            )

            container_name = f"sepilot-{instance_id}-{os.getpid()}".replace("__", "-")

            # 기존 동명 컨테이너 제거
            try:
                old = client.containers.get(container_name)
                old.remove(force=True)
            except docker.errors.NotFound:
                pass

            # 컨테이너 내 bootstrap 명령:
            #
            # SWE-bench 컨테이너 Python 환경 구조:
            #   - base conda (/opt/miniconda3/bin/python): Python 3.11+ → 에이전트 실행 가능
            #   - testbed conda (/opt/miniconda3/envs/testbed/bin/python): Python 3.6~3.10
            #     → 프로젝트 의존성(django/numpy/sympy 등) 설치되어 있음
            #
            # 실행 순서:
            # 1. base conda Python으로 에이전트 venv 생성 (langchain 등 agent 의존성 설치)
            # 2. venv 생성 완료 후 PATH 앞에 testbed bin 추가
            #    → 이후 subprocess 호출 시 `python` = testbed Python (프로젝트 패키지 있음)
            # 3. agent는 전용 venv Python(명시적 경로)으로 실행
            #    → agent 자체는 base Python 3.11 환경
            # 4. agent가 bash_execute로 `python script.py` 실행 시
            #    → PATH 상 testbed Python이 먼저 → 프로젝트 패키지 정상 import ✅
            team_flag = "--team-mode " if self.team_mode else ""
            bootstrap_cmd = (
                # [1] base conda Python(3.11+)으로 에이전트 전용 venv 생성
                f"python -m venv {self.CONTAINER_AGENT_VENV} 2>/dev/null && "
                f"{self.CONTAINER_AGENT_VENV}/bin/pip install --quiet "
                "rich pydantic pyyaml click httpx "
                "langchain langchain-community langchain-openai langgraph "
                "python-dotenv html2text 2>&1 | tail -5; "
                # [1.5] testbed Python에 pytest 설치 (에이전트가 테스트 실행 시 사용)
                "/opt/miniconda3/envs/testbed/bin/pip install --quiet pytest 2>&1 | tail -2; "
                # [2] PATH 앞에 testbed bin 추가 (이 시점부터 `python` = testbed Python)
                "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && "
                "cd /testbed && "
                # [3] agent는 agent venv Python으로 명시적 실행
                "PYTHONPATH=/sepilot "
                f"{self.CONTAINER_AGENT_VENV}/bin/python -m sepilot.agent.bench.run_in_container "
                f"--instance-id {instance_id} "
                f"--problem-file {self.CONTAINER_PROBLEM_FILE} "
                f"--output-path {self.CONTAINER_RESULT_FILE} "
                f"{team_flag}"
            )

            container = client.containers.run(
                image=image_key,
                name=container_name,
                command=["bash", "-c", bootstrap_cmd],
                volumes={
                    str(self.project_dir): {"bind": "/sepilot", "mode": "ro"},
                    str(problem_file): {"bind": self.CONTAINER_PROBLEM_FILE, "mode": "ro"},
                    str(self.OUTPUT_DIR): {"bind": self.CONTAINER_OUTPUT_DIR, "mode": "rw"},
                },
                environment=env_vars,
                network_mode="host",
                working_dir="/testbed",
                detach=True,
            )

            # 컨테이너 실행 대기
            timed_out = False
            try:
                exit_result = container.wait(timeout=timeout)
                exit_code = exit_result.get("StatusCode", -1)
            except Exception as wait_err:
                logs.append(f"Container wait error (timeout): {wait_err}")
                exit_code = -1
                timed_out = True
                # SIGTERM 먼저 → 시그널 핸들러가 result.json 작성할 시간 확보
                try:
                    container.kill(signal="SIGTERM")
                    import time as _time
                    _time.sleep(3)  # result.json 작성 대기
                except Exception:
                    pass
                # 아직 실행 중이면 강제 종료
                try:
                    container.kill()
                except Exception:
                    pass

            # 로그 수집 (충분한 양의 로그 확보)
            try:
                container_logs = container.logs(tail=2000).decode("utf-8", errors="replace")
                logs.append(f"Container logs:\n{container_logs[-20000:]}")

                # 전체 로그를 별도 파일로 저장
                log_file = self.OUTPUT_DIR / f"logs_{instance_id}.txt"
                log_file.write_text(container_logs, encoding="utf-8")
            except Exception:
                pass

            # 결과 수집 - 컨테이너 내부에서 result.json 복사
            model_patch = ""
            try:
                # 컨테이너 내부 result.json 읽기
                bits, _ = container.get_archive(self.CONTAINER_RESULT_FILE)
                # tar 아카이브에서 파일 추출
                import io
                import tarfile
                tar_stream = io.BytesIO()
                for chunk in bits:
                    tar_stream.write(chunk)
                tar_stream.seek(0)
                with tarfile.open(fileobj=tar_stream) as tar:
                    member = tar.getmembers()[0]
                    f = tar.extractfile(member)
                    if f:
                        container_result = json.loads(f.read().decode("utf-8"))
                        model_patch = container_result.get("model_patch", "")
                        if container_result.get("error"):
                            logs.append(f"Agent error: {container_result['error']}")

                        # 에이전트 세션 로그 추출 및 저장
                        agent_log = container_result.get("agent_session_log", "")
                        if agent_log:
                            agent_log_file = self.OUTPUT_DIR / f"agent_log_{instance_id}.jsonl"
                            agent_log_file.write_text(agent_log, encoding="utf-8")
                            logs.append(f"Agent session log saved: {agent_log_file}")
            except Exception as e:
                logs.append(f"Result collection error: {e}")

            # 폴백: git diff를 get_archive로 추출 (stopped 컨테이너에서도 가능)
            if not model_patch:
                model_patch = self._extract_git_diff_via_archive(container, logs)

            # 컨테이너 정리
            try:
                container.remove(force=True)
            except Exception:
                pass

            # 임시 파일 정리
            try:
                problem_file.unlink(missing_ok=True)
            except Exception:
                pass

            duration = time.time() - start_time
            logs.append(f"Exit code: {exit_code}")
            logs.append(f"Completed in {duration:.2f}s")
            logs.append(f"Patch length: {len(model_patch)}")

            bench_result = BenchResult(
                instance_id=instance_id,
                status="success" if model_patch else "no_patch",
                model_patch=model_patch if model_patch else None,
                test_result="pending",
                duration_seconds=duration,
                logs="\n".join(logs),
                error_message=None if model_patch else "No patch generated",
            )

            self._save_result(bench_result)
            return bench_result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logs.append(f"Error: {error_msg}")
            self.console.print(f"[red]{instance_id}: {error_msg}[/red]")

            return BenchResult(
                instance_id=instance_id,
                status="error",
                error_message=error_msg,
                duration_seconds=duration,
                logs="\n".join(logs),
            )

    def _build_env_vars(self) -> dict[str, str]:
        """컨테이너에 전달할 환경변수 구성."""
        env_vars = {}

        # LLM API 설정 + 멀티모델 tier 환경변수
        for key in [
            "OPENAI_API_BASE_URL", "OPENAI_API_KEY", "DEFAULT_MODEL",
            "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "LLM_BASE_URL", "API_BASE_URL", "OLLAMA_BASE_URL",
            "SEPILOT_TRIAGE_MODEL", "SEPILOT_VERIFIER_MODEL",
            "SEPILOT_REASONING_MODEL", "SEPILOT_QUICK_MODEL",
        ]:
            val = os.getenv(key)
            if val:
                env_vars[key] = val

        env_vars["PYTHONPATH"] = "/sepilot"

        if self.team_mode:
            env_vars["TEAM_MODE"] = "1"

        return env_vars

    def run_batch(
        self,
        instances: list[SWEInstance],
        max_workers: int = 1,
        timeout: int = 600,
    ) -> list[BenchResult]:
        """배치 실행 (순차 또는 병렬)."""
        results = []

        self.console.print(
            f"\n[cyan]Running inference for {len(instances)} instances "
            f"(workers={max_workers}, timeout={timeout}s)...[/cyan]"
        )

        if max_workers <= 1:
            for i, inst in enumerate(instances, 1):
                self.console.print(
                    f"\n[bold][{i}/{len(instances)}] {inst.instance_id}[/bold]"
                )
                result = self.run_instance(inst, timeout=timeout)
                results.append(result)

                status_color = "green" if result.model_patch else "red"
                patch_len = len(result.model_patch) if result.model_patch else 0
                self.console.print(
                    f"  [{status_color}]{result.status}[/{status_color}] "
                    f"(patch: {patch_len} chars, {result.duration_seconds:.1f}s)"
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_inst = {
                    executor.submit(self.run_instance, inst, timeout): inst
                    for inst in instances
                }
                for future in as_completed(future_to_inst):
                    inst = future_to_inst[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = BenchResult(
                            instance_id=inst.instance_id,
                            status="error",
                            error_message=str(e),
                        )
                    results.append(result)
                    self.console.print(
                        f"  [{'green' if result.model_patch else 'red'}]"
                        f"{inst.instance_id}: {result.status}[/]"
                    )

        return results

    def save_predictions(
        self,
        results: list[BenchResult],
        output_path: Path | None = None,
    ) -> Path:
        """predictions.jsonl 저장 (swebench 호환 형식)."""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.PREDICTIONS_DIR / f"predictions_{timestamp}.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions = []
        for r in results:
            pred = Prediction(
                instance_id=r.instance_id,
                model_patch=r.model_patch or "",
            )
            predictions.append(pred)

        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred.model_dump(), ensure_ascii=False) + "\n")

        self.console.print(f"[green]Predictions saved: {output_path}[/green]")
        self.console.print(
            f"  [dim]{len(predictions)} predictions, "
            f"{sum(1 for p in predictions if p.model_patch)} with patches[/dim]"
        )

        # latest 심볼릭 링크
        latest_link = self.PREDICTIONS_DIR / "predictions_latest.jsonl"
        try:
            latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(output_path.name)
        except Exception:
            pass

        return output_path

    def _extract_git_diff_via_archive(self, container, logs: list) -> str:
        """정지된 컨테이너에서 get_archive로 git diff 추출.

        exec_run은 stopped 컨테이너에서 동작하지 않으므로,
        임시 스크립트를 통해 git diff를 파일로 저장한 뒤 get_archive로 추출.
        실패 시 빈 문자열 반환.
        """
        import io
        import tarfile

        # 방법 1: 컨테이너를 잠시 재시작하여 git diff 실행
        try:
            container.start()
            exec_result = container.exec_run(
                ["bash", "-c", f"cd /testbed && git diff HEAD > {self.CONTAINER_PATCH_FILE}"],
                workdir="/testbed",
            )
            # git diff 파일 추출
            bits, _ = container.get_archive(self.CONTAINER_PATCH_FILE)
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            with tarfile.open(fileobj=tar_stream) as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f:
                    patch = f.read().decode("utf-8", errors="replace").strip()
                    if patch:
                        logs.append(f"Fallback: extracted git diff via restart ({len(patch)} chars)")
                        # 컨테이너 다시 정지
                        try:
                            container.kill()
                        except Exception:
                            pass
                        return patch
        except Exception as e:
            logs.append(f"Fallback git diff via restart failed: {e}")

        # 방법 2: .git 디렉토리에서 직접 diff 추출은 불가능하므로,
        # 변경된 파일 목록이라도 로그에 기록
        try:
            container.kill()
        except Exception:
            pass

        return ""

    def _save_result(self, result: BenchResult) -> None:
        """개별 결과를 JSON 파일로 저장."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.OUTPUT_DIR / f"{timestamp}_{result.instance_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)


class EvaluationRunner:
    """Phase 2: swebench.harness.run_evaluation으로 패치 평가."""

    BASE_WORK_DIR = Path(tempfile.gettempdir()) / "sepilot-bench"
    EVAL_RESULTS_DIR = BASE_WORK_DIR / "evaluation_results"

    def __init__(self, logger, console: Console):
        self.logger = logger
        self.console = console
        self.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        predictions_path: str | Path,
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "test",
        instance_ids: list[str] | None = None,
        max_workers: int = 4,
        timeout: int = 1800,
        run_id: str | None = None,
    ) -> dict:
        """swebench.harness.run_evaluation.main() Python API로 평가 실행.

        Returns:
            평가 결과 딕셔너리 {instance_id: {resolved: bool, ...}}
        """
        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        # predictions 에서 instance_ids 추출 (지정되지 않은 경우)
        if instance_ids is None:
            instance_ids = []
            with open(predictions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        instance_ids.append(data["instance_id"])

        if not run_id:
            run_id = f"sepilot_{time.strftime('%Y%m%d_%H%M%S')}"

        self.console.print(
            f"[cyan]Evaluating {len(instance_ids)} predictions with swebench harness...[/cyan]"
        )
        self.console.print(f"  [dim]Predictions: {predictions_path}[/dim]")
        self.console.print(f"  [dim]Run ID: {run_id}[/dim]")

        try:
            self._run_swebench_eval(
                dataset_name=dataset_name,
                split=split,
                instance_ids=instance_ids,
                predictions_path=str(predictions_path),
                max_workers=max_workers,
                run_id=run_id,
                timeout=timeout,
            )
        except Exception as e:
            self.console.print(f"[red]Evaluation error: {e}[/red]")
            raise

        # 결과 파일 찾기 및 파싱
        results = self._collect_results(run_id, instance_ids)

        return results

    def _run_swebench_eval(
        self,
        dataset_name: str,
        split: str,
        instance_ids: list[str],
        predictions_path: str,
        max_workers: int,
        run_id: str,
        timeout: int,
    ) -> None:
        """swebench 평가를 python3 subprocess로 실행.

        swebench는 시스템 python3에만 설치되어 있으므로,
        직접 import 대신 subprocess로 python3를 호출합니다.
        """
        # python3 경로 확인 (swebench가 설치된 파이썬)
        python3 = shutil.which("python3") or "/usr/bin/python3"

        eval_script = f"""
import json, sys
from swebench.harness.run_evaluation import main as run_evaluation

instance_ids = json.loads('{json.dumps(instance_ids)}')

run_evaluation(
    dataset_name="{dataset_name}",
    split="{split}",
    instance_ids=instance_ids,
    predictions_path="{predictions_path}",
    max_workers={max_workers},
    force_rebuild=False,
    cache_level="none",
    clean=False,
    open_file_limit=4096,
    run_id="{run_id}",
    timeout={timeout},
    namespace=None,
    rewrite_reports=False,
    modal=False,
    instance_image_tag="latest",
    env_image_tag="latest",
    report_dir="{self.EVAL_RESULTS_DIR}",
)
"""
        self.console.print(f"  [dim]Using {python3} for swebench evaluation[/dim]")

        result = subprocess.run(
            [python3, "-c", eval_script],
            capture_output=True,
            text=True,
            timeout=timeout * len(instance_ids) + 600,  # 총 타임아웃
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                self.console.print(f"  [dim]{line}[/dim]")

        if result.returncode != 0:
            stderr_tail = "\n".join(result.stderr.strip().split("\n")[-5:]) if result.stderr else ""
            raise RuntimeError(f"swebench evaluation failed (exit {result.returncode}): {stderr_tail}")

    def _collect_results(
        self, run_id: str, instance_ids: list[str]
    ) -> dict:
        """swebench harness 평가 결과 수집.

        swebench 결과 파일 형식:
        {
            "resolved_ids": ["instance1", "instance2", ...],
            "unresolved_ids": ["instance3", ...],
            "resolved_instances": 20,
            "total_instances": 29,
            ...
        }
        """
        results = {}

        # swebench 결과 파일 검색 (여러 위치/패턴)
        result_file = self._find_result_file(run_id)

        if result_file and result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                raw_results = json.load(f)

            # swebench 보고서 형식: resolved_ids / unresolved_ids 리스트
            resolved_ids = set(raw_results.get("resolved_ids", []))

            if resolved_ids or "resolved_ids" in raw_results:
                # 신규 형식 (resolved_ids 리스트)
                for instance_id in instance_ids:
                    results[instance_id] = {
                        "resolved": instance_id in resolved_ids,
                        "raw": raw_results,
                    }
            else:
                # 레거시 형식 (instance_id → {resolved: bool})
                for instance_id in instance_ids:
                    if instance_id in raw_results:
                        entry = raw_results[instance_id]
                        resolved = entry.get("resolved", False) if isinstance(entry, dict) else False
                        results[instance_id] = {"resolved": resolved, "raw": entry}
                    else:
                        results[instance_id] = {"resolved": False, "raw": {}}

            # 결과 파일을 EVAL_RESULTS_DIR에 보관
            dest = self.EVAL_RESULTS_DIR / result_file.name
            try:
                if result_file != dest:
                    shutil.copy2(result_file, dest)
            except Exception:
                pass

            resolved_count = sum(1 for r in results.values() if r.get("resolved"))
            self.console.print(
                f"\n[bold]Evaluation Results:[/bold]\n"
                f"  Total: {len(results)}\n"
                f"  [green]Resolved: {resolved_count}[/green]\n"
                f"  [red]Unresolved: {len(results) - resolved_count}[/red]\n"
                f"  Pass rate: {resolved_count / len(results) * 100:.1f}%"
            )
        else:
            self.console.print(
                "[yellow]No evaluation result file found. "
                "Check swebench harness output.[/yellow]"
            )
            for instance_id in instance_ids:
                results[instance_id] = {"resolved": False, "raw": {}}

        return results

    def _find_result_file(self, run_id: str) -> Path | None:
        """swebench 결과 파일을 여러 위치/패턴에서 검색."""
        # swebench는 report_dir에 "sepilot.{run_id}.json" 형식으로 저장
        candidates = [
            self.EVAL_RESULTS_DIR / f"sepilot.{run_id}.json",
            self.EVAL_RESULTS_DIR / f"{run_id}.json",
            Path(f"sepilot.{run_id}.json"),
            Path(f"results/{run_id}.json"),
        ]

        for path in candidates:
            if path.exists():
                return path

        # 글로브 검색 (run_id 부분 매칭)
        for search_dir in [self.EVAL_RESULTS_DIR, Path(".")]:
            for f in search_dir.glob(f"*{run_id}*.json"):
                return f

        return None
