#!/usr/bin/env python3
"""Clean VSCode cell markers from .py and .ipynb files in the workspace.

Usage: python clean_vscode_cells.py [path]
If no path is given, uses the current directory.
"""
import sys
import json
from pathlib import Path

MARKERS = ("<VSCode.Cell", "</VSCode.Cell")


def clean_py(path: Path) -> int:
    changed = 0
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    new_lines = [l for l in lines if not any(m in l for m in MARKERS)]
    if new_lines != lines:
        backup = path.with_suffix(path.suffix + ".bak")
        path.rename(backup)
        path.write_text("".join(new_lines), encoding="utf-8")
        print(f"Cleaned {path} (backup -> {backup})")
        changed = 1
    return changed


def clean_ipynb(path: Path) -> int:
    changed = 0
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    modified = False
    for cell in cells:
        src = cell.get("source")
        if not src:
            continue
        # source can be a list of strings or a single string
        if isinstance(src, list):
            new_src = [s for s in src if not any(m in s for m in MARKERS)]
            if new_src != src:
                cell["source"] = new_src
                modified = True
        elif isinstance(src, str):
            lines = src.splitlines(keepends=True)
            new_lines = [l for l in lines if not any(m in l for m in MARKERS)]
            new_src = "".join(new_lines)
            if new_src != src:
                cell["source"] = new_src
                modified = True
    if modified:
        backup = path.with_suffix(path.suffix + ".bak")
        path.rename(backup)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Cleaned {path} (backup -> {backup})")
        changed = 1
    return changed


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    total = 0
    for p in sorted(root.rglob("*.py")):
        # skip this script's backup or itself if desired
        if p.name == Path(__file__).name:
            continue
        total += clean_py(p)
    for p in sorted(root.rglob("*.ipynb")):
        total += clean_ipynb(p)
    print(f"Done. Files modified: {total}")


if __name__ == "__main__":
    main()
