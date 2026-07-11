#!/usr/bin/env python3
"""Seed data/gt_contacts.json from analysis outputs and rebuild pose prototypes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gt_contacts import seed_gt_contacts_from_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output/ directory",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip rebuilding phase_prototypes.json",
    )
    args = parser.parse_args()
    result = seed_gt_contacts_from_analysis(
        output_root=args.output_root,
        rebuild=not args.no_rebuild,
    )
    print(
        f"Done. total_samples={result['total']} "
        f"counts={result['counts']} skipped={result['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
