#!/usr/bin/env python3
"""Repo-wide one-way-data-flow / layering gate (CI-blocking).

Why this exists
---------------
The architecture has a strict, load-bearing one-way separation
(``pipeline/`` -> ``models/`` -> ``apps/api/*/services.py`` -> views -> web).
The ``.claude/hooks/layering_tripwire.py`` hook guards only *Claude-driven*
edits; this script guards **every** commit (the CI enforcement layer), so the
invariant holds regardless of how an edit was made.

Invariants enforced
--------------------
1. **Views must route through services.py** — never import the model layer
   directly. Any ``apps/api/*/views.py`` line matching
   ``^\\s*(from|import)\\s+models\\b`` is a violation.

2. **models/ and shared/ are the pure-Python serving layer** — no Django, no
   Django app imports. Any ``.py`` under ``models/`` or ``shared/`` with a line
   matching ``^\\s*(from|import)\\s+(django|apps)\\b`` is a violation.

Stdlib only. Line-oriented regex (intentionally — it mirrors the hook and
catches the import forms that matter without an AST round-trip). Exit 1 on any
violation, printed as ``file:line``.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root = two levels up from scripts/ci/check_layering.py.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Rule 1: views.py must not import the model layer.
#   from models...  /  import models...
_VIEWS_FORBIDDEN = re.compile(r"^\s*(from|import)\s+models\b")

# Rule 2: models/ and shared/ must not import django or the django apps.
#   from django... / import django... / from apps... / import apps...
_PURE_FORBIDDEN = re.compile(r"^\s*(from|import)\s+(django|apps)\b")


def _iter_lines(path: Path):
    """Yield (lineno, text) for a source file, tolerant of encoding issues."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    for i, line in enumerate(text.splitlines(), start=1):
        yield i, line


def _check_views() -> list[str]:
    violations: list[str] = []
    api_dir = REPO_ROOT / "apps" / "api"
    if not api_dir.is_dir():
        return violations
    for vf in sorted(api_dir.glob("*/views.py")):
        if "migrations" in vf.parts:
            continue
        for lineno, line in _iter_lines(vf):
            if _VIEWS_FORBIDDEN.match(line):
                rel = vf.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel}:{lineno}: views.py imports `models` directly "
                    f"(must route through services.py) -> {line.strip()}"
                )
    return violations


def _check_pure_layers() -> list[str]:
    violations: list[str] = []
    for top in ("models", "shared"):
        base = REPO_ROOT / top
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            for lineno, line in _iter_lines(py):
                if _PURE_FORBIDDEN.match(line):
                    rel = py.relative_to(REPO_ROOT)
                    violations.append(
                        f"{rel}:{lineno}: pure layer imports django/apps "
                        f"(must stay framework-free) -> {line.strip()}"
                    )
    return violations


def main() -> int:
    violations = _check_views() + _check_pure_layers()

    if violations:
        print("LAYERING GATE FAILED:")
        for v in violations:
            print(f"  {v}")
        print()
        print(f"check_layering: {len(violations)} violation(s).")
        return 1

    print("check_layering: OK — no one-way-flow violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
