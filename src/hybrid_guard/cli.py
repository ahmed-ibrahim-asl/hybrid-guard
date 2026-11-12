"""Command line: ``hybrid-guard check predictions.csv --target y --incumbent base --candidate hybrid``.

Reads a CSV or Parquet file, runs CMAT for every ``--candidate`` column against the
incumbent column, prints the summaries and optionally writes an HTML report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hybrid_guard.cmat import cmat
from hybrid_guard.report import save_report


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hybrid-guard", description="Coverage-matched acceptance test for hybrid models.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="test one or more candidate columns against an incumbent column")
    check.add_argument("file", type=Path)
    check.add_argument("--target", required=True, help="column with the measured values")
    check.add_argument("--incumbent", required=True, help="column with the incumbent model's predictions")
    check.add_argument("--candidate", required=True, action="append",
                       help="candidate column; NAME=COLUMN to rename; repeat for several candidates")
    check.add_argument("--selected", action="append", default=[],
                       help="CANDIDATE_NAME=BOOLEAN_COLUMN marking selected rows for a selective policy")
    check.add_argument("--time", help="timestamp column (rows are resampled in whole blocks)")
    check.add_argument("--group", help="block column, used instead of --time")
    check.add_argument("--block", default="24h", help="block length for --time (default 24h)")
    check.add_argument("--materiality", type=float, default=0.01)
    check.add_argument("--harm-bound", type=float, default=0.10)
    check.add_argument("--margin", type=float, default=0.002)
    check.add_argument("--min-rows", type=int, default=30)
    check.add_argument("--replicates", type=int, default=10_000)
    check.add_argument("--html", type=Path, help="write an HTML report to this path")
    check.add_argument("--title", default="Acceptance report")
    args = parser.parse_args(argv)

    frame = _read(args.file)
    selections = dict(item.split("=", 1) for item in args.selected)
    results = {}
    for item in args.candidate:
        name, column = item.split("=", 1) if "=" in item else (item, item)
        selected = frame[selections[name]].astype(bool).to_numpy() if name in selections else None
        results[name] = cmat(frame[args.target], frame[args.incumbent], frame[column], selected,
                             timestamps=frame[args.time] if args.time else None,
                             groups=frame[args.group] if args.group else None, block=args.block,
                             materiality=args.materiality, harm_bound=args.harm_bound, margin=args.margin,
                             min_rows=args.min_rows, replicates=args.replicates)
        print(f"[{name}]\n{results[name].summary()}\n")
    if args.html:
        save_report(args.html, results, title=args.title, dataset=f"{args.file.name}: incumbent {args.incumbent}")
        print(f"report written to {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
