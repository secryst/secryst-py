#!/usr/bin/env python3
"""Parity check: regenerate outputs from the committed zip and diff
against the committed goldens. Exit 1 on any mismatch. This is the
Python leg of the cross-crystal parity gate; Ruby/TS CI run the same
golden.jsonl against the same tiny-1.0.zip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from secryst import Model  # noqa: E402

MAX_LEN = 8


def main() -> int:
    goldens = [json.loads(l) for l in (HERE / "golden.jsonl").read_text().splitlines()]
    model = Model(HERE / "tiny-1.0.zip")
    bad = 0
    for row in goldens:
        got = model.translate(row["input"], max_len=MAX_LEN)
        status = "ok" if got == row["output"] else "MISMATCH"
        if got != row["output"]:
            bad += 1
        print(f"[{status}] {row['input']!r} -> {got!r} (golden {row['output']!r})")
    print(f"{'PASS' if bad == 0 else 'FAIL'}: {len(goldens) - bad}/{len(goldens)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
