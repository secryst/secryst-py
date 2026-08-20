#!/usr/bin/env python3
"""Generate the cross-crystal parity kit.

Builds a tiny IMF v1 zip whose decoder graph deterministically emits a
constant token every step (argmax pinned via a broadcast bias), forcing
a real greedy decode loop of `MAX_LEN` steps per input. No training, no
randomness: every conforming runtime MUST produce byte-identical
outputs. Runs the Python reference crystal over a fixed multilingual
input list and writes golden.jsonl.

The zip + goldens are committed so Ruby/TypeScript CI consume the same
artifacts without building anything.

Usage: python3 parity/generate_goldens.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from secryst import Model  # noqa: E402

VOCAB = 260                 # covers byte+3 for every byte value
EMIT = ord("A") + 3         # constant argmax token -> 'A'
MAX_LEN = 8

MANIFEST = {
    "format": "imf-v1",
    "id": "parity-tiny-1.0",
    "task": "parity",
    "tokenizer": "bytes",
    "opset": 14,
    "decoder": "plain",
    "precision": "fp32",
    "license": "BSD-3-Clause",
    "trained_from": "cross-crystal parity fixture (synthetic graphs)",
}

INPUTS = ["A", "hello", "ភាសា", "قَدَر", "ספר", "ไทย", "你好", ""]


def _opset(ir: int = 8):
    m = helper.make_model(
        helper.make_graph([], "empty", [], []), opset_imports=[helper.make_opsetid("", 14)]
    )
    m.ir_version = ir
    return m


def _encoder() -> bytes:
    # last_hidden_state = input_ids + 0 (int64 passthrough)
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input_ids", "zero"], ["last_hidden_state"])],
        name="parity-enc",
        inputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.INT64, ["batch", "seq"]
            )
        ],
        initializer=[
            numpy_helper.from_array(np.zeros(1, dtype=np.int64), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    return model.SerializeToString()


def _decoder() -> bytes:
    # logits[1, seq, VOCAB] = 0 + bias[VOCAB]; bias[EMIT] = 1.0
    # -> argmax over the last dim is EMIT at every step: the greedy loop
    # runs MAX_LEN real steps and decodes 'A' * MAX_LEN for any input.
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Shape", ["input_ids"], ["seq_shape"]),
            helper.make_node("Gather", ["seq_shape", "one"], ["seq_dim"], axis=0),
            helper.make_node(
                "Concat", ["b_one", "seq_dim", "v_dim"], ["logit_shape"], axis=0
            ),
            helper.make_node(
                "ConstantOfShape", ["logit_shape"], ["zeros"],
                value=helper.make_tensor("", TensorProto.FLOAT, [1], [0.0]),
            ),
            helper.make_node("Add", ["zeros", "bias"], ["logits"]),
        ],
        name="parity-dec",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info(
                "encoder_hidden_states", TensorProto.INT64, ["batch", "seq"]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["batch", "seq", "vocab"]
            )
        ],
        initializer=[
            numpy_helper.from_array(np.array([1], dtype=np.int64), "b_one"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
            numpy_helper.from_array(np.array([VOCAB], dtype=np.int64), "v_dim"),
            numpy_helper.from_array(
                np.eye(1, VOCAB, EMIT, dtype=np.float32).reshape(VOCAB), "bias"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    return model.SerializeToString()


def build_parity_zip(path: Path) -> Path:
    enc, dec = _encoder(), _decoder()
    meta = dict(MANIFEST)
    meta["sha256"] = {
        "encoder.onnx": hashlib.sha256(enc).hexdigest(),
        "decoder.onnx": hashlib.sha256(dec).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump(meta))
        zf.writestr("encoder.onnx", enc)
        zf.writestr("decoder.onnx", dec)
    return path


def main() -> None:
    zip_path = build_parity_zip(HERE / "tiny-1.0.zip")
    model = Model(zip_path)
    with open(HERE / "golden.jsonl", "w", encoding="utf-8") as f:
        for text in INPUTS:
            row = {"input": text, "output": model.translate(text, max_len=MAX_LEN)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{text!r} -> {row['output']!r}")


if __name__ == "__main__":
    main()
