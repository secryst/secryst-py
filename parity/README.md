# Cross-crystal parity kit

`tiny-1.0.zip` — an IMF v1 fixture with synthetic ONNX graphs whose
decoder argmax is pinned to a constant token, forcing a real greedy
loop of 8 steps per input. Fully deterministic; no training.

`golden.jsonl` — outputs of the Python reference crystal over a fixed
multilingual input list. Every crystal MUST reproduce these
byte-for-byte (interscript-ml v1, conformance requirement C5).

- Python: `python3 parity/verify.py`
- Ruby: `bundle exec rspec spec/parity_spec.rb` (in secryst/secryst,
  CI checks out this repo for the fixture + goldens)
- Regenerate after fixture changes: `python3 parity/generate_goldens.py`
