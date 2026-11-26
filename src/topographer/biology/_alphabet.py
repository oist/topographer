"""Immutable ordered Alphabet and biological presets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Alphabet:
    """Immutable ordered alphabet with unique single-character symbols.

    Examples
    --------
    >>> from topographer.biology import Alphabet
    >>> alpha = Alphabet(("A", "C", "G", "T"))
    >>> "A" in alpha
    True
    """

    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("Alphabet cannot be empty.")
        seen = set()
        for s in self.symbols:
            if not isinstance(s, str) or len(s) != 1:
                raise ValueError(f"Alphabet symbol {s!r} must be a single character.")
            if s in seen:
                raise ValueError(f"Duplicate alphabet symbol {s!r}.")
            seen.add(s)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbols

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self):
        return iter(self.symbols)

    def validate_sequence(self, sequence: str) -> None:
        """Validate that sequence contains only symbols from this alphabet."""
        if not sequence:
            raise ValueError("Sequence cannot be empty.")
        unknown = {ch for ch in sequence if ch not in self.symbols}
        if unknown:
            raise ValueError(f"Sequence contains undeclared symbols: {sorted(unknown)}")


class _PresetAlphabet(Alphabet):
    """Subclass to enable docstrings on preset instances."""


DNA_ALPHABET = Alphabet(("A", "C", "G", "T"))
"""Standard DNA alphabet (A, C, G, T).

Examples
--------
>>> from topographer.biology import DNA_ALPHABET
>>> len(DNA_ALPHABET)
4
"""

RNA_ALPHABET = Alphabet(("A", "C", "G", "U"))
"""Standard RNA alphabet (A, C, G, U).

Examples
--------
>>> from topographer.biology import RNA_ALPHABET
>>> len(RNA_ALPHABET)
4
"""

AMINO_ACIDS_20 = Alphabet(
    (
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    )
)
"""Canonical 20-amino-acid alphabet.

Examples
--------
>>> from topographer.biology import AMINO_ACIDS_20
>>> len(AMINO_ACIDS_20)
20
"""

AMINO_ACIDS_21 = Alphabet(
    (
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "*",
    )
)
"""21-symbol amino-acid alphabet with generic '*' stop.

Examples
--------
>>> from topographer.biology import AMINO_ACIDS_21
>>> "*" in AMINO_ACIDS_21
True
"""

AMINO_ACIDS_23 = Alphabet(
    (
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "%",
        "^",
        "&",
    )
)
"""Topographer 23-symbol amino-acid alphabet with resolved stop codons.

Examples
--------
>>> from topographer.biology import AMINO_ACIDS_23
>>> "%" in AMINO_ACIDS_23
True
"""
