from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_scalar(text: str) -> Any:
    t = text.strip()
    if t == "":
        return ""
    low = t.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    try:
        if any(ch in t for ch in [".", "e", "E"]):
            return float(t)
        return int(t)
    except ValueError:
        return t


def _set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _load_yaml_builtin(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1

        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise RuntimeError(f"invalid yaml list item near: {raw!r}")
            parent.append(_parse_scalar(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            raise RuntimeError(f"invalid yaml line: {raw!r}")

        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()

        if val == "":
            j = i
            next_nonempty = ""
            next_indent = -1
            while j < len(lines):
                nxt = lines[j]
                j += 1
                s = nxt.strip()
                if not s or s.startswith("#"):
                    continue
                next_nonempty = s
                next_indent = len(nxt) - len(nxt.lstrip(" "))
                break

            if next_nonempty.startswith("- ") and next_indent > indent:
                node: Any = []
            else:
                node = {}

            if not isinstance(parent, dict):
                raise RuntimeError(f"yaml parent is not mapping near: {raw!r}")
            parent[key] = node
            stack.append((indent, node))
            continue

        if not isinstance(parent, dict):
            raise RuntimeError(f"yaml parent is not mapping near: {raw!r}")
        if "#" in val:
            val = val.split("#", 1)[0].rstrip()
        parent[key] = _parse_scalar(val)

    return root


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        if not isinstance(obj, dict):
            raise RuntimeError("config root must be mapping")
        return obj
    except ModuleNotFoundError:
        obj = _load_yaml_builtin(text)
        if not isinstance(obj, dict):
            raise RuntimeError("config root must be mapping")
        return obj


def load_config(path: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    cfg_path = Path(path)
    cfg = _load_yaml(cfg_path)

    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise RuntimeError(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        _set_path(out, key.strip(), _parse_scalar(raw.strip()))
    return out
