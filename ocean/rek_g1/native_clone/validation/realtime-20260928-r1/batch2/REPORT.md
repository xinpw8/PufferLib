# Batch2 controller compatibility

The one-arena controller is a reasonable numerical-compatibility candidate for the real-time app test. Its coefficients are unchanged. This result does not certify bit-identical controller or physical trajectories.

| Measurement | Values compared | Unequal FP32 bits | Maximum absolute difference |
|---|---:|---:|---:|
| Quantized encoder tokens | 4,096 | 0 | 0 |
| Decoder, identical raw inputs | 1,856 | 1,740 | 0.00000905990600586 |
| Connected encoder and decoder | 1,856 | 1,774 | 0.00000953674316406 |
| Decoder, identical batch8 tokens | 1,856 | 1,774 | 0.00000953674316406 |

All 32 deterministic fixture pairs used the same first two encoder inputs and decoder histories in batch2 and batch8. The other six batch8 rows were distinct. All 96 decoder comparisons were within the previously used absolute 0.00002 decoder criterion. That criterion comes from the earlier native-versus-ORT test included under `prior-criterion`; it is applied here as a bounded compatibility judgment. The exact comparison still failed with exit code 1, as preserved in the original closed result. No tolerance or output was changed.

The CPU audit compared all 44 initializer tensors (22,523,613 float values) exactly, together with 388 constant float values. The only allowed constant differences were 20 leading int64 batch dimensions, 2 versus 8. The original receipt and an independently repeated CPU-only run agree. These coefficient checks alone do not prove equality of arbitrary ONNX graph semantics; the native reader and the direct inference test provide the narrower execution evidence.

The connected comparison found no larger discrepancy than the decoder with identical tokens. This supports a small decoder arithmetic difference on these samples; it does not establish the precise CUDA arithmetic cause. Untested encoder quantization thresholds and contact-sensitive physical trajectories remain possible sources of divergent future behavior. No Unity or official-server physics, fighting performance, or general closed-loop equivalence is claimed.

Deployment still requires the complete app to sustain 50 control steps/s with the unchanged 20 ms control interval and ten 2 ms physics substeps. One arena represents the interactive two-fighter task. CUDA graph execution remains disabled.

`source/build.sh` documents the exact CUDA SM121 build and pins the reused Sonic object. The two ARM64 test binaries are included; model weights and the external Sonic object are excluded and identified by paths and hashes. `source/analyze_closed.py` can reproduce `ASSESSMENT.json` measurements from the original closed JSONL without CUDA.
