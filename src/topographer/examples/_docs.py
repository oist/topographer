"""Generate the committed documentation assets from deterministic examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

from topographer.examples._cases import CASES, export_case

__all__ = [
    "DocsAssetsResult",
    "write_all_docs_assets",
]


@dataclass(frozen=True, slots=True)
class DocsAssetsResult:
    """Paths to exported documentation assets."""

    cube_paths: dict[str, Path]
    hypercube_paths: dict[str, Path]

    @property
    def table_paths(self) -> dict[str, Path]:
        return {
            k.removeprefix("table_"): v
            for k, v in self.hypercube_paths.items()
            if k.startswith("table_")
        }

    @property
    def chart_paths(self) -> dict[str, dict[str, Path]]:
        charts: dict[str, dict[str, Path]] = {}
        for k, v in self.hypercube_paths.items():
            if k.startswith("chart_"):
                rest = k.removeprefix("chart_")
                name, _, fmt = rest.rpartition("_")
                charts.setdefault(name, {})[fmt] = v
                if "simplified" in name:
                    charts.setdefault("simplification", {})[fmt] = v
        return charts


def write_all_docs_assets(output_directory: str | Path = "docs") -> DocsAssetsResult:
    """Regenerate curated Gallery tables and figures under the documentation tree from the example workspaces."""
    output_path = Path(output_directory)
    figures_root = output_path / "figures"
    interactive_root = output_path / "interactive"
    data_root = output_path / "data"

    results = {}
    for name, case in CASES.items():
        fig_dir = figures_root / name
        interactive_dir = interactive_root / name
        data_dir = data_root / name
        fig_dir.mkdir(parents=True, exist_ok=True)
        interactive_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        exported = export_case(case, case.build_topology(), fig_dir)

        # Move CSV tables from figures/<name>/ to data/<name>/
        for csv_file in fig_dir.glob("*.csv"):
            dest_file = data_dir / csv_file.name
            if dest_file.exists():
                dest_file.unlink()
            shutil.move(str(csv_file), str(dest_file))

        # Move HTML interactive charts from figures/<name>/ to interactive/<name>/
        for html_file in fig_dir.glob("*.html"):
            dest_html = interactive_dir / html_file.name
            if dest_html.exists():
                dest_html.unlink()
            shutil.move(str(html_file), str(dest_html))

        # Update path references in exported
        resolved_exported = {}
        for k, p in exported.items():
            if p.suffix == ".csv":
                resolved_exported[k] = data_dir / p.name
            elif p.suffix == ".html":
                resolved_exported[k] = interactive_dir / p.name
            else:
                resolved_exported[k] = fig_dir / p.name
        results[name] = resolved_exported

    return DocsAssetsResult(
        cube_paths=results["cube"],
        hypercube_paths=results["hypercube"],
    )


def main() -> None:
    """Regenerate every committed documentation asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    arguments = parser.parse_args()
    write_all_docs_assets(arguments.output_dir)


if __name__ == "__main__":
    main()
