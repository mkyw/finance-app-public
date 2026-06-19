#!/usr/bin/env python3.11
"""SessionStart advisory: nudge when terminal investigations are unconsolidated.

Read-only, pure stdlib. Scans ``agent-artifacts/investigations/`` for active
investigation files (loose ``*.md``, excluding the digest/workflow docs and the
``archive/`` subfolder) whose header region looks *terminal* — fix shipped or
hypothesis refuted/misdiagnosed — but that are not yet recorded in ``MASTER.md``.

If any are found, prints the Claude Code ``SessionStart`` hook JSON so the advisory
reaches the model's context once at session start; otherwise exits 0 silently.

This is a conservative advisory only: it scans the header region (first lines) to
keep false positives low and never decides on its own. The authoritative call —
which investigations to consolidate, and how — belongs to the ``inv-consolidate``
agent. Wired in ``.claude/settings.json`` as a ``SessionStart`` hook.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Repo root: prefer the harness-provided project dir, else derive from this file.
_REPO_ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", "")) or Path(__file__).resolve().parents[1]
_INV_DIR = _REPO_ROOT / "agent-artifacts" / "investigations"
_MASTER = _INV_DIR / "MASTER.md"

# Non-investigation files in the folder.
_SKIP_NAMES = {"MASTER.md", "WORKFLOW.md", "README.md"}

# Header region scanned for status markers (status lives near the top).
_HEADER_LINES = 20

# Terminal-status markers. Anchored to the status vocabulary these headers
# actually use — NOT bare prose verbs ("built"/"shipped"/"✅" appear in active
# scoping docs too and over-fire). Two tiers: status phrases (any case) and
# all-caps standalone status tokens (case-sensitive, so prose "reversed"/
# "resolved" don't trip it).
_TERMINAL_PHRASE_RE = re.compile(
    r"prediction-affecting build|misdiagnos|refuted|no defect|superseded", re.IGNORECASE
)
_TERMINAL_CAPS_RE = re.compile(r"\b(REVERSED|RESOLVED|BUILT)\b")
# Active-status markers that veto a terminal match (still has work to do).
_ACTIVE_RE = re.compile(r"deferred|awaiting|\bSTOP\b|gated|scoping|buildability", re.IGNORECASE)


def _stale_count() -> int:
    if not _INV_DIR.is_dir():
        return 0
    master_text = _MASTER.read_text(encoding="utf-8") if _MASTER.exists() else ""

    count = 0
    for path in sorted(_INV_DIR.glob("*.md")):
        if path.name in _SKIP_NAMES:
            continue
        slug = path.stem
        if slug in master_text:  # already consolidated/recorded
            continue
        try:
            header = "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:_HEADER_LINES])
        except OSError:
            continue
        is_terminal = _TERMINAL_PHRASE_RE.search(header) or _TERMINAL_CAPS_RE.search(header)
        if is_terminal and not _ACTIVE_RE.search(header):
            count += 1
    return count


def main() -> int:
    n = _stale_count()
    if n > 0:
        msg = (
            f"⚠ {n} investigation(s) in agent-artifacts/investigations/ look terminal "
            f"and unconsolidated; consider running the inv-consolidate agent to fold them "
            f"into agent-artifacts/investigations/MASTER.md."
        )
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": msg,
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
