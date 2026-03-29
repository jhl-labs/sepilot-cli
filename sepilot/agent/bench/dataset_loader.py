import json
import logging
from pathlib import Path

from pydantic import ValidationError

from sepilot.agent.bench.models import SWEInstance

logger = logging.getLogger(__name__)


class DatasetLoader:
    @staticmethod
    def _parse_instance(data: object, source: str) -> SWEInstance:
        if not isinstance(data, dict):
            raise ValueError(f"{source}: expected object, got {type(data).__name__}")
        try:
            return SWEInstance(**data)
        except ValidationError as e:
            raise ValueError(f"{source}: invalid SWEInstance payload: {e}") from e

    @staticmethod
    def load_predefined_dataset(dataset_config: dict) -> list[SWEInstance]:
        from datasets import load_dataset

        hf_name = dataset_config.get("dataset_name", "princeton-nlp/SWE-bench")
        hf_revision = dataset_config.get("dataset_revision", "main")
        swebench = load_dataset(
            hf_name,
            split=dataset_config["split"],
            revision=hf_revision,
        )

        requested_ids = set(dataset_config["instance_ids"])
        filtered = [
            item
            for item in swebench
            if item["instance_id"] in requested_ids
        ]

        # Warn about missing instances
        found_ids = {item["instance_id"] for item in filtered}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            logger.warning(
                "Preset '%s': %d/%d instances not found in %s: %s",
                dataset_config.get("name", "unknown"),
                len(missing_ids),
                len(requested_ids),
                hf_name,
                sorted(missing_ids),
            )

        instances = []
        for item in filtered:
            data = dict(item)
            if isinstance(data.get("FAIL_TO_PASS"), str):
                data["FAIL_TO_PASS"] = json.loads(data["FAIL_TO_PASS"])
            if isinstance(data.get("PASS_TO_PASS"), str):
                data["PASS_TO_PASS"] = json.loads(data["PASS_TO_PASS"])
            instances.append(DatasetLoader._parse_instance(data, "predefined dataset item"))

        return instances

    @staticmethod
    def load_from_file(file_path: str) -> list[SWEInstance]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        instances = []

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        instances.append(
                            DatasetLoader._parse_instance(
                                data,
                                f"line {line_num}",
                            )
                        )
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        raise ValueError(f"Failed to parse line: {e}") from e
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data_list = json.load(f)
            if isinstance(data_list, list):
                instances = [
                    DatasetLoader._parse_instance(item, f"item {idx}")
                    for idx, item in enumerate(data_list, 1)
                ]
            elif isinstance(data_list, dict):
                instances = [DatasetLoader._parse_instance(data_list, "root object")]
            else:
                raise ValueError(
                    f"Unsupported JSON dataset shape: {type(data_list).__name__}. "
                    "Use an object or list of objects."
                )
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. Use .json or .jsonl"
            )

        return instances
