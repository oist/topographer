"""CLI entry point for exporting the topographer example workspaces."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from topographer.examples._cases import CASES, export_case, load_or_build_workspace


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or reload an example workspace and export its tables and figures. "
            "The workspace is saved under <output-directory>/workspace."
        )
    )
    parser.add_argument("case", choices=tuple(CASES), help="Example case to export.")
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        default=None,
        help="Output directory for the workspace, tables and figures (default: output/<case>).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the workspace even if a saved one exists in the output directory.",
    )
    parser.add_argument(
        "--simplification-threshold",
        type=float,
        default=None,
        help="Tree simplification persistence threshold (default: the case's own).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    case = CASES[args.case]
    output_dir = args.output_directory or Path("output") / args.case
    topology = load_or_build_workspace(case, output_dir / "workspace", force=args.force)
    export_case(
        case,
        topology,
        output_dir,
        simplification_threshold=args.simplification_threshold,
    )


if __name__ == "__main__":
    main()
