"""Private polygonal-face representation used by sign-epistasis counting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Face:
    """An ordered polygonal 2-cell boundary given as a cycle of vertex indices.

    Attributes
    ----------
    vertices:
        Three or more distinct vertex indices forming a simple cycle.
    face_id:
        Optional identifier for this face.
    payload:
        Optional metadata (for example ``face_kind`` of ``"square"`` or ``"triangle"``).
    """

    vertices: tuple[int, ...]
    face_id: Any = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError(f"A face must have at least 3 vertices, got {len(self.vertices)}.")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError(f"Face vertices must be distinct simple cycles, got {self.vertices}.")
        object.__setattr__(self, "vertices", tuple(int(v) for v in self.vertices))
