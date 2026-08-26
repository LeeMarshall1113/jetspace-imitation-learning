#!/usr/bin/env python3
"""Everything that should pass before a large merge lands on main.

    python scripts/premerge_check.py

A textually clean merge is not a working one. The feature branch carries 50
commits written against the OLD EpisodeBuffer API while main carries PR #9's
replacement, and git resolves that silently because the two sides touched
different files. So the checks here are the ones git cannot do:

  1. every file parses
  2. every module imports (catches API drift at import time)
  3. no caller uses the retired EpisodeBuffer API
  4. every relative link in the README and docs resolves to a real file
  5. the round-trip tests pass

Run against a merged worktree, not against either parent.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Retired by PR #9. A caller still using these merged cleanly but would fail
#: at runtime, which is the whole risk this script exists to catch.
RETIRED = [
    (re.compile(r"\blen\(\s*buf(?:fer)?\s*\)"), "len(buffer) -- use buffer.buffer_size()"),
    (re.compile(r"\bbuf(?:fer)?\.success\b"), "buffer.success -- use buffer.episode.success"),
    (re.compile(r"\bbuf(?:fer)?\.action\b"), "buffer.action -- use buffer.episode.action"),
    (re.compile(r"\bbuf(?:fer)?\.proprio\b"), "buffer.proprio -- use buffer.episode.proprio"),
    (re.compile(r"\bbuf(?:fer)?\.pixels\b"), "buffer.pixels -- use buffer.episode.pixels"),
    (re.compile(r"\.Episode\b"), ".Episode -- renamed to .episode"),
]

failures: list[str] = []


def check_parses() -> None:
    bad = []
    for p in list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("scripts/*.py")) \
            + list(ROOT.glob("tests/*.py")):
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(ROOT)}:{e.lineno} {e.msg}")
    print(f"1. parse            {'OK' if not bad else f'{len(bad)} FAILED'}")
    for b in bad:
        print(f"     {b}")
    failures.extend(bad)


def check_retired_api() -> None:
    hits = []
    me = Path(__file__).resolve()
    for p in list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("scripts/*.py")):
        # This file defines the very patterns being searched for.
        if p.resolve() == me:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pat, msg in RETIRED:
                if pat.search(line):
                    hits.append(f"{p.relative_to(ROOT)}:{i} {msg}")
    print(f"2. retired API      {'OK' if not hits else f'{len(hits)} FOUND'}")
    for h in hits:
        print(f"     {h}")
    failures.extend(hits)


def check_imports() -> None:
    """Import every package module. Scripts are skipped: many execute work at
    import time or need a GPU, and importing them here would run experiments."""
    bad = []
    sys.path.insert(0, str(ROOT / "src"))
    for p in sorted(ROOT.glob("src/jetspace/**/*.py")):
        if p.name == "__init__.py":
            continue
        mod = ".".join(p.relative_to(ROOT / "src").with_suffix("").parts)
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{mod}: {type(e).__name__}: {str(e)[:80]}")
    print(f"3. imports          {'OK' if not bad else f'{len(bad)} FAILED'}")
    for b in bad:
        print(f"     {b}")
    failures.extend(bad)


def check_links() -> None:
    """Relative markdown links must resolve. A README describing results whose
    evidence is not on this branch is worse than a stale one."""
    pat = re.compile(r"\[[^\]]+\]\(([^)#][^)]*)\)")
    bad = []
    for md in [ROOT / "README.md", *sorted(ROOT.glob("docs/*.md"))]:
        if not md.exists():
            continue
        for m in pat.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1).split("#")[0].strip()
            if not target or target.startswith(("http", "mailto:")):
                continue
            resolved = (md.parent / target).resolve()
            # Links like ../../issues escape the repo and are resolved by
            # GitHub against the repository URL, not the filesystem.
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue
            if not resolved.exists():
                bad.append(f"{md.relative_to(ROOT)} -> {target}")
    print(f"4. markdown links   {'OK' if not bad else f'{len(bad)} BROKEN'}")
    for b in bad[:20]:
        print(f"     {b}")
    if len(bad) > 20:
        print(f"     ... and {len(bad) - 20} more")
    failures.extend(bad)


def check_tests() -> None:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    last = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:] or ["no output"]
    print(f"5. tests            {'OK' if r.returncode == 0 else 'FAILED'}  {last[0]}")
    if r.returncode != 0:
        failures.append("pytest failed")


def main() -> int:
    print("=" * 70)
    print(f"pre-merge check: {ROOT}")
    print("=" * 70)
    check_parses()
    check_retired_api()
    check_imports()
    check_links()
    check_tests()
    print("=" * 70)
    if failures:
        print(f"{len(failures)} problem(s). Do not merge until these are clear.")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
