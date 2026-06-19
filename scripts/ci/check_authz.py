#!/usr/bin/env python3
"""Mechanical API-authorization gate (CI-blocking).

Why this exists
---------------
CI is the *only* enforcement layer that guards every edit path — not just
Claude-driven edits. The repo sets **no** ``REST_FRAMEWORK`` defaults, so a DRF
view that forgets ``permission_classes`` silently falls back to DRF's library
default (``AllowAny`` in this configuration) — an unauthenticated-access footgun
that no test would necessarily catch. This script makes that mistake fail the
build.

What it checks
--------------
Walks ``apps/api/*/views.py`` (skipping migrations) and, for every class-based
DRF view (a class whose bases reference ``APIView`` / ``generics.*`` /
``viewsets.*`` / the ``*View`` / ``*ViewSet`` families), enforces:

  1. FAIL if the class has **no** ``permission_classes`` attribute at all
     (implicit default — a footgun).
  2. FAIL if ``permission_classes`` is **empty** (``[]`` / ``()``) AND the
     view's dotted path (``<app>.views.<ClassName>``) is **not** listed in
     ``scripts/ci/authz_public_allowlist.txt``.
  3. PASS otherwise.

Stdlib only (``ast`` based — no Django import, no DB, no settings). Exit 1 on
any violation with a clear per-view listing.

A NEW deliberately-public endpoint is unblocked by adding its dotted path to
the allowlist *with a justification comment* — see that file's header. The
``/new-endpoint`` skill and the ``authz`` audit agent both reference this gate.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Repo root = two levels up from this file (scripts/ci/check_authz.py).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
API_DIR = REPO_ROOT / "apps" / "api"
ALLOWLIST_PATH = Path(__file__).resolve().parent / "authz_public_allowlist.txt"

# Apps excluded from the gate. Empty — the gate is strict everywhere. (The legacy
# core/ app, whose unauthenticated UserViewSet was the original exemption, was
# removed in the fix(security) commit; keep this mechanism only for a future
# genuinely-transitional app, never as a parking spot for an exposed endpoint.)
_EXEMPT_APPS: frozenset[str] = frozenset()

# Base-class name fragments that mark a class as a DRF view. Matched against the
# last dotted component of each base (so ``generics.ListAPIView``,
# ``viewsets.ModelViewSet``, ``APIView``, a project ``BaseAPIView`` mixin, etc.
# all qualify). Kept deliberately broad — a false positive is a harmless extra
# check; a false negative is an ungated public endpoint.
_VIEW_BASE_MARKERS = ("APIView", "GenericAPIView", "ViewSet", "View")
# Explicit DRF generic view class names (these end in ...APIView/...View and are
# already covered by the markers above, but listed for clarity/intent).


def _base_name(node: ast.expr) -> str:
    """Return the trailing identifier of a base-class expression.

    ``APIView`` -> ``APIView``; ``generics.ListAPIView`` -> ``ListAPIView``;
    ``rest_framework.views.APIView`` -> ``APIView``. Subscripted/other exotic
    bases return ""."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _looks_like_view(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        name = _base_name(base)
        if any(name.endswith(marker) for marker in _VIEW_BASE_MARKERS):
            return True
    return False


def _find_permission_classes(class_node: ast.ClassDef) -> ast.expr | None:
    """Return the RHS node of a class-body ``permission_classes = ...``.

    Returns ``None`` when the attribute is absent. Handles both plain
    assignment (``permission_classes = [...]``) and annotated assignment
    (``permission_classes: list = [...]`` — the form used in this repo).
    """
    for stmt in class_node.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
            value = stmt.value  # may be None for a bare annotation
        else:
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == "permission_classes":
                return value
    return None


def _is_empty_sequence(node: ast.expr | None) -> bool:
    """True for ``[]`` or ``()`` (the deliberately-public marker)."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts) == 0
    return False


def _load_allowlist() -> set[str]:
    if not ALLOWLIST_PATH.exists():
        return set()
    entries: set[str] = set()
    for raw in ALLOWLIST_PATH.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


def _dotted_path(views_file: Path, class_name: str) -> str:
    """``apps/api/profiles/views.py`` + ``CityResolveView`` ->
    ``profiles.views.CityResolveView`` (the app-rooted dotted path, matching how
    the repo's apps are installed: ``INSTALLED_APPS = [..., "profiles", ...]``)."""
    app = views_file.parent.name  # e.g. "profiles"
    return f"{app}.views.{class_name}"


def main() -> int:
    if not API_DIR.is_dir():
        print(f"check_authz: apps/api not found at {API_DIR}", file=sys.stderr)
        return 1

    allowlist = _load_allowlist()
    views_files = sorted(
        p
        for p in API_DIR.glob("*/views.py")
        if "migrations" not in p.parts and p.parent.name not in _EXEMPT_APPS
    )
    if _EXEMPT_APPS:
        print(
            f"check_authz: NOTE — exempting legacy app(s) pending removal: "
            f"{', '.join(sorted(_EXEMPT_APPS))} (see _EXEMPT_APPS comment).\n"
        )

    failures: list[str] = []
    checked = 0

    for vf in views_files:
        try:
            tree = ast.parse(vf.read_text(), filename=str(vf))
        except SyntaxError as exc:  # pragma: no cover - syntax gate catches this
            failures.append(f"{vf}: SyntaxError: {exc}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not _looks_like_view(node):
                continue

            checked += 1
            dotted = _dotted_path(vf, node.name)
            perm = _find_permission_classes(node)

            if perm is None:
                # No permission_classes at all -> implicit default footgun.
                failures.append(
                    f"FAIL {dotted}: no `permission_classes` attribute "
                    f"(implicit DRF default is unauthenticated-open; set it "
                    f"explicitly)"
                )
                print(f"FAIL {dotted}: missing permission_classes")
            elif _is_empty_sequence(perm):
                if dotted in allowlist:
                    print(
                        f"PASS {dotted}: empty permission_classes — "
                        f"allowlisted (deliberately public)"
                    )
                else:
                    failures.append(
                        f"FAIL {dotted}: empty `permission_classes` (public) "
                        f"but not in {ALLOWLIST_PATH.name}. If this endpoint is "
                        f"intentionally public, add `{dotted}` to that file "
                        f"with a justification comment; otherwise set real "
                        f"permission_classes."
                    )
                    print(f"FAIL {dotted}: public but not allowlisted")
            else:
                print(f"PASS {dotted}: permission_classes set")

    print()
    print(
        f"check_authz: inspected {checked} DRF view(s) across "
        f"{len(views_files)} views.py file(s); "
        f"{len(failures)} violation(s)."
    )

    if failures:
        print()
        print("AUTHZ GATE FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("check_authz: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
