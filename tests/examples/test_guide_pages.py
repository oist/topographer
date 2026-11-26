"""The Python blocks of the user-guide pages run top to bottom."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "guide"


@pytest.mark.parametrize("page", ["bootstrap-persistence-band.qmd"])
def test_guide_page_code_runs(page: str) -> None:
    text = (GUIDE / page).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks
    exec(compile("\n".join(blocks), page, "exec"), {"__name__": "guide"})  # noqa: S102
