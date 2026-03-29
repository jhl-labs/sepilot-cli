"""PresetManager for team preset management (builtin / global / project)."""

from __future__ import annotations

import copy
import logging
import os
import shlex
import shutil
from dataclasses import replace
from pathlib import Path

import yaml

from sepilot.agent.multi.models import AgentRole, Strategy, TeamPreset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in presets (hardcoded)
# ---------------------------------------------------------------------------

_BUILTIN_PRESETS: dict[str, TeamPreset] = {
    "dev-team": TeamPreset(
        name="dev-team",
        description="개발 팀",
        strategy=Strategy.AUTO,
        roles=[
            AgentRole(name="developer", agent_cmd="claude", system_prompt="코드를 구현하세요."),
            AgentRole(name="tester", agent_cmd="claude", system_prompt="테스트를 작성하세요."),
            AgentRole(name="reviewer", agent_cmd="claude", system_prompt="코드를 리뷰하세요."),
        ],
    ),
    "full-team": TeamPreset(
        name="full-team",
        description="전체 팀",
        strategy=Strategy.AUTO,
        roles=[
            AgentRole(name="architect", agent_cmd="claude", system_prompt="아키텍처를 설계하세요."),
            AgentRole(name="developer", agent_cmd="claude", system_prompt="코드를 구현하세요."),
            AgentRole(name="tester", agent_cmd="claude", system_prompt="테스트를 작성하세요."),
            AgentRole(name="reviewer", agent_cmd="claude", system_prompt="코드를 리뷰하세요."),
        ],
    ),
    "qa-team": TeamPreset(
        name="qa-team",
        description="품질 보증 팀",
        strategy=Strategy.PARALLEL,
        roles=[
            AgentRole(name="tester", agent_cmd="claude", system_prompt="테스트를 작성하세요."),
            AgentRole(
                name="security-auditor",
                agent_cmd="claude",
                system_prompt="보안 취약점을 점검하세요.",
            ),
        ],
    ),
}


def _normalize_strategy(value: object) -> str:
    """Return a valid strategy value, falling back to AUTO for malformed input."""
    try:
        return Strategy(value or Strategy.AUTO).value
    except (TypeError, ValueError):
        logger.warning("알 수 없는 프리셋 strategy '%s', auto로 대체합니다.", value)
        return Strategy.AUTO.value


def _preset_to_dict(preset: TeamPreset) -> dict:
    """Serialize a TeamPreset to a YAML-friendly dict."""
    roles_list = []
    for r in preset.roles:
        role_dict: dict = {
            "name": r.name,
            "agent": r.agent_cmd,
            "prompt": r.system_prompt,
        }
        if getattr(r, "workdir", None):
            role_dict["workdir"] = r.workdir
        roles_list.append(role_dict)

    return {
        "name": preset.name,
        "description": preset.description,
        "strategy": _normalize_strategy(preset.strategy),
        "roles": roles_list,
    }


def _dict_to_preset(name: str, d: dict) -> TeamPreset:
    """Deserialize a dict (from YAML) into a TeamPreset."""
    raw_roles = d.get("roles", [])
    if raw_roles is None:
        raw_roles = []
    elif not isinstance(raw_roles, list):
        raise ValueError(f"프리셋 '{name}'의 roles가 list가 아닙니다.")

    roles = []
    for idx, raw_role in enumerate(raw_roles):
        if not isinstance(raw_role, dict):
            logger.warning("프리셋 '%s'의 roles[%d] 항목이 dict가 아닙니다.", name, idx)
            continue

        role_name = raw_role.get("name")
        if not isinstance(role_name, str) or not role_name.strip():
            logger.warning("프리셋 '%s'의 roles[%d] name이 올바르지 않습니다.", name, idx)
            continue

        agent_cmd = raw_role.get("agent", "claude")
        if not isinstance(agent_cmd, str) or not agent_cmd.strip():
            agent_cmd = "claude"

        system_prompt = raw_role.get("prompt", "")
        if not isinstance(system_prompt, str):
            system_prompt = str(system_prompt or "")

        workdir = raw_role.get("workdir")
        if workdir is not None and not isinstance(workdir, str):
            workdir = str(workdir)

        roles.append(
            AgentRole(
                name=role_name.strip(),
                agent_cmd=agent_cmd,
                system_prompt=system_prompt,
                workdir=workdir,
            )
        )

    if raw_roles and not roles:
        raise ValueError(f"프리셋 '{name}'에 유효한 roles가 없습니다.")

    return TeamPreset(
        name=name,
        description=d.get("description", ""),
        strategy=_normalize_strategy(d.get("strategy", Strategy.AUTO)),
        roles=roles,
    )


def _validate_agents(preset: TeamPreset) -> TeamPreset:
    """Return a copy with role.available set based on shutil.which()."""
    new_roles = []
    for role in preset.roles:
        # 플래그/인용부호 포함 명령에서 실제 실행 파일만 추출한다.
        agent_bin = role.agent_cmd
        if role.agent_cmd:
            try:
                parts = shlex.split(role.agent_cmd, posix=os.name != "nt")
                agent_bin = parts[0] if parts else ""
            except ValueError:
                logger.warning("에이전트 명령 파싱 실패, 원본 문자열로 판정합니다: %s", role.agent_cmd)
                agent_bin = role.agent_cmd.strip().split()[0] if role.agent_cmd.strip() else ""
        avail = shutil.which(agent_bin) is not None
        if not avail:
            logger.warning(
                "에이전트 '%s' (역할 '%s')를 찾을 수 없습니다.",
                role.agent_cmd,
                role.name,
            )
        new_roles.append(replace(role, available=avail))
    return replace(preset, roles=new_roles)


class PresetManager:
    """Manages team presets from three layers: builtin > global > project.

    Priority (load_preset): project > global > builtin.
    """

    def __init__(
        self,
        global_dir: str | None = None,
        project_file: str | None = None,
    ) -> None:
        if global_dir is None:
            global_dir = os.path.join(
                os.path.expanduser("~"), ".sepilot", "agent_presets"
            )
        self._global_dir = Path(global_dir)
        self._global_dir.mkdir(parents=True, exist_ok=True)
        self._project_file = Path(project_file) if project_file else None
        # Deep-copy builtins so we never mutate them
        self._builtins: dict[str, TeamPreset] = copy.deepcopy(_BUILTIN_PRESETS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_presets(self) -> list[TeamPreset]:
        """Return all presets: builtin + global + project (project overrides)."""
        merged: dict[str, TeamPreset] = {}

        # Layer 1: builtins
        for name, preset in self._builtins.items():
            merged[name] = preset

        # Layer 2: global directory
        for preset in self._load_global_presets():
            merged[preset.name] = preset

        # Layer 3: project file (highest priority)
        for preset in self._load_project_presets():
            merged[preset.name] = preset

        return [
            _validate_agents(copy.deepcopy(preset))
            for preset in merged.values()
        ]

    def load_preset(self, name: str) -> TeamPreset | None:
        """Load a preset by name. Priority: project > global > builtin.

        Validates agent availability on the returned copy.
        """
        # Try project first
        proj = self._load_project_preset(name)
        if proj is not None:
            return _validate_agents(proj)

        # Try global
        gp = self._load_global_preset(name)
        if gp is not None:
            return _validate_agents(gp)

        # Try builtin (return a copy so builtins are never mutated)
        bp = self._builtins.get(name)
        if bp is not None:
            return _validate_agents(copy.deepcopy(bp))

        return None

    def save_preset(self, name: str, preset: TeamPreset) -> None:
        """Save a preset to the global directory as YAML (원자적 쓰기)."""
        import tempfile

        path = self._global_dir / f"{name}.yaml"
        data = _preset_to_dict(preset)
        # 임시 파일에 쓴 뒤 원자적 교체 (부분 쓰기 방지)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._global_dir), suffix=".yaml.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            os.replace(tmp_path, str(path))
        except Exception:
            # 실패 시 임시 파일 정리
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def delete_preset(self, name: str) -> None:
        """Delete a preset from the global directory. No-op if not found."""
        path = self._global_dir / f"{name}.yaml"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_global_presets(self) -> list[TeamPreset]:
        """Load all presets from the global directory."""
        results = []
        if not self._global_dir.exists():
            return results
        for p in sorted(self._global_dir.glob("*.yaml")):
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    name = data.get("name", p.stem)
                    results.append(_dict_to_preset(name, data))
            except Exception:
                logger.warning("글로벌 프리셋 파일 로드 실패: %s", p)
        return results

    def _load_global_preset(self, name: str) -> TeamPreset | None:
        """Load a single preset from the global directory."""
        path = self._global_dir / f"{name}.yaml"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                return _dict_to_preset(data.get("name", name), data)
        except Exception:
            logger.warning("글로벌 프리셋 파일 로드 실패: %s", path)
        return None

    def _load_project_presets(self) -> list[TeamPreset]:
        """Load all presets from the project-level agents.yaml."""
        data = self._read_project_file()
        if not data:
            return []
        results = []
        for name, preset_data in data.items():
            try:
                results.append(_dict_to_preset(name, preset_data))
            except Exception:
                logger.warning("프로젝트 프리셋 로드 실패: %s", name)
        return results

    def _load_project_preset(self, name: str) -> TeamPreset | None:
        """Load a single preset from the project-level agents.yaml."""
        data = self._read_project_file()
        if not data or name not in data:
            return None
        try:
            return _dict_to_preset(name, data[name])
        except Exception:
            logger.warning("프로젝트 프리셋 로드 실패: %s", name)
            return None

    def _read_project_file(self) -> dict | None:
        """Read and return the presets dict from the project YAML, or None."""
        if self._project_file is None or not self._project_file.exists():
            return None
        try:
            with open(self._project_file, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if raw and isinstance(raw, dict):
                presets = raw.get("presets", {})
                if presets is None:
                    return {}
                if isinstance(presets, dict):
                    return presets
                logger.warning(
                    "프로젝트 프리셋 파일의 presets 섹션이 dict가 아닙니다: %s",
                    self._project_file,
                )
                return {}
        except Exception:
            logger.warning("프로젝트 프리셋 파일 로드 실패: %s", self._project_file)
        return None
