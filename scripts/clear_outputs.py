"""
Clear pipeline output folders for a fresh run.

Deletes the contents of vast_export/ (logs, CSVs, SWCs, meshes).
Pass -d / --diag to also clear diag_output/.

Usage:
    python scripts/clear_outputs.py
    python scripts/clear_outputs.py -d
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPORT_DIR  = REPO_ROOT / "vast_export"
DIAG_DIR    = REPO_ROOT / "diag_output"


def _remove_dir_contents(path: Path) -> int:
    """Delete everything inside path without removing path itself. Returns item count."""
    if not path.exists():
        print(f"  (skip) {path} does not exist")
        return 0
    count = 0
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear pipeline output folders.")
    parser.add_argument(
        "-d", "--diag",
        action="store_true",
        help="Also clear diag_output/",
    )
    args = parser.parse_args()

    targets = [EXPORT_DIR]
    if args.diag:
        targets.append(DIAG_DIR)

    print("The following items will be removed:")
    any_exist = False
    for path in targets:
        if not path.exists():
            print(f"  (skip) {path}  -- does not exist")
            continue
        for item in sorted(path.iterdir()):
            kind = "dir " if item.is_dir() else "file"
            print(f"  [{kind}]  {item}")
            any_exist = True

    if not any_exist:
        print("  (nothing to remove)")
        return

    print()
    answer = input("Confirm? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    print()
    for path in targets:
        n = _remove_dir_contents(path)
        print(f"  cleared {path.relative_to(REPO_ROOT)}  ({n} item(s) removed)")

    print("Done.")


if __name__ == "__main__":
    main()
