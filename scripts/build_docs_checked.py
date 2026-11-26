"""Build documentation and reject unexpected great-docs warnings."""

from __future__ import annotations

import re
import subprocess
import sys


def main() -> int:
    command = ["great-docs", "build"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdout is not None
    warnings = []
    for line in process.stdout:
        print(line, end="", flush=True)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if "[!!]" in plain or re.search(r"\bWARN(?:ING)?\b", plain, re.IGNORECASE):
            if (
                "Dynamic introspection failed for a resolved item" in plain
                or "API reference generated (static analysis)" in plain
                or re.search(r"\b\d+ warnings?\b", plain, re.IGNORECASE)
                or "Warning: _quarto.yml not found. Creating minimal configuration" in plain
            ):
                continue
            warnings.append(plain.rstrip())
    status = process.wait()
    if warnings:
        print("Unexpected documentation warnings:", *warnings, sep="\n", file=sys.stderr)
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
