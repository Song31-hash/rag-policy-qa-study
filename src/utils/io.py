"""JSON/YAML 입출력 및 경로 유틸."""
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def get_project_root() -> Path:
    """프로젝트 루트 (scripts/ 또는 src/ 의 상위)."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    """YAML 설정 로드."""
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: Path) -> None:
    """YAML 저장."""
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_json(path: Path) -> Any:
    """JSON 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """JSON 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """중첩 dict를 재귀적으로 merge. override가 우선."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_with_base(path: Path, project_root: Path | None = None) -> dict[str, Any]:
    """YAML 로드 후 _base_ 있으면 해당 파일 로드해 deep merge."""
    data = load_yaml(path)
    if not data:
        return {}

    base_ref = data.pop("_base_", None)
    if not base_ref:
        return data

    root = project_root or get_project_root()

    # 1순위: 현재 config 파일 기준 상대경로
    base_path = (path.parent / base_ref).resolve()

    # 2순위: 프로젝트 루트 기준 상대경로
    if not base_path.exists():
        base_path = (root / base_ref).resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_ref}")

    base = load_config_with_base(base_path, project_root=root)
    return _deep_merge(base, data)


def resolve_paths(config: dict, project_root: Path | None = None) -> dict:
    """config 내 paths 키를 project_root 기준 절대 경로로 변환."""
    root = project_root or get_project_root()

    if "paths" not in config or not isinstance(config["paths"], dict):
        return config

    out = dict(config)
    resolved_paths = {}

    for key, value in config["paths"].items():
        if isinstance(value, str):
            p = Path(value)
            resolved_paths[key] = p if p.is_absolute() else (root / p)
        else:
            resolved_paths[key] = value

    out["paths"] = resolved_paths
    return out