# Native live policy worker validation

The persistent JSONL worker loads the selected V4 mixed checkpoint and performs
sampled BF16 CUDA inference without environment stepping. It is a transport and
inference component; these tests do not demonstrate authentic-game observation
equivalence, autonomous fighting, or parity.

- CPU-only parser build: 109 assertions, 35 responses, zero actions, exit 0.
- GPU build on Spark: 121 assertions, 35 responses, five masked actions, exit 0.
- GPU synthetic request latency: 0.114097 ms minimum, 0.176066 ms median,
  0.370836 ms maximum. Five requests are a smoke test, not a throughput study.
- Startup identity and callback shutdown tests: four Node tests passed.
- No Python, Torch, ONNX Runtime, or MuJoCo dependency in the native worker.

The GPU test used synthetic zero observations and one-hot masks. Tests cover
strict dimensions, finite values, mask values, no legal action, malformed JSON,
duplicate/unknown fields, stale sequence numbers, line-size limit, explicit
reset, measured round change, and action-free terminal acknowledgment.

Checkpoint SHA256:
`0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b`

Worker SHA256:
`3a12e88268bca78ccec4813adaeacaec74bd68652594b46007d24ab711c0a029`

The linked `native_policy.o` is the exact existing build-v4 object, SHA256
`4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c`.

[Summary](summary.json), [CPU test transcript](protocol-test.stdout.json),
[GPU test transcript](gpu-test.stdout.json), [build hashes](build-hashes.txt),
[dependencies](elf-dependencies.txt), and [protocol documentation](../../LIVE_POLICY_WORKER.md).

No checkpoint, model, proprietary game binary, authentic observation array,
credential, or live input is included in this directory.
