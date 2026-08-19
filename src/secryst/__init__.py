"""interscript-ml — the Python runtime for Interscript Model Format (IMF v1).

The reference implementation: the Ruby and TypeScript runtimes are
diffed against this one on shared golden sets.

    from secryst import Model
    model = Model.load("khm-latn-1.0.zip")
    model.translate("ភាសា")        # -> "pheasaea"

Byte-level only: the tokenizer is the canonical ByT5 table (byte b ->
token id b+3, trailing EOS), fixed and documented — no vocab files.
"""

from __future__ import annotations

from secryst.loader import Manifest, ModelFormatError
from secryst.model import Model
from secryst.registry import RegistryError, resolve
from secryst.tokens import BYTE_OFFSET, EOS_ID, PAD_ID, UNK_ID, decode, encode

__all__ = [
    "BYTE_OFFSET",
    "EOS_ID",
    "Manifest",
    "Model",
    "ModelFormatError",
    "PAD_ID",
    "RegistryError",
    "UNK_ID",
    "decode",
    "encode",
    "resolve",
]
