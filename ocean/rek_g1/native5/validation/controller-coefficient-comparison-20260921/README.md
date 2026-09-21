# Alternate-copy controller coefficients

Offline comparison completed 2026-09-21. **SAME learned coefficients** as the public GEAR-SONIC controller already used by the full physical backend. No inference, GPU use, training, deployment, or model replacement was performed.

The exact alternate-copy TextAssets were privately extracted from `C:/rekagent/rek-agent-build/REK_Data/sharedassets1.assets`, using existing UnityPy 1.25.2 tooling. Path IDs 186/187/188 are the decoder, configuration, and encoder identified in the [producer-boundary investigation](../reward-objective-20260919/balance-producer-boundary.md). Container source-before/source-after SHA256 remained `a47d6fc85303142975ae825bb5fb7a893d2026efdadc4600e5252fe17547cd9b`.

| Extracted input | Bytes | SHA256 |
|---|---:|---|
| Decoder | 40,900,688 | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| Configuration | 11,880 | `9e8e18d763adfdce094ece2061a42acf4b48e43f89b62ccea3d8b7ba25c2a355` |
| Encoder | 50,100,513 | `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3` |

## Direct comparison

A standalone C++ inspector used the unchanged native `sonic_onnx_reader.h`, with no graph executor. It compared decoded tensor names, dimensions, types, and every coefficient byte against both public originals and the actual batch-8 physical-backend graphs. The conclusion therefore does not depend on interpreting different serialized-file hashes as different weights.

| Tensor group | Compared | Result |
|---|---:|---|
| Encoder learned initializers | 30 tensors, 12,339,392 FP32 coefficients | All identical |
| Decoder learned initializers | 14 tensors, 10,184,221 FP32 coefficients | All identical |
| Encoder constants versus public original | 97 tensors | All identical |
| Decoder constants versus either reference | 15 tensors | All identical |
| Encoder constants versus batch 8 | 77 identical, 20 different | Each difference is only the leading int64 1 becoming 8 |

All 388 FP32 Constant coefficients, including quantizer constants, match. No learned tensor is missing or additional. All six graphs pass the existing native parser, exact I/O and operator-count checks, required layer names/shapes/finite coefficient extraction, and quantizer checks at the appropriate batch size. Four inspector self-checks cover identity, changed coefficient, changed shape, and equivalent raw/repeated-float storage. Final CPU inspection took 0.49 s and exited 0.

The original graphs are batch 1. The existing matching batch-8 graphs remain the usable inputs for the four-arena physical backend. Extracting this old copy has not produced a controller with different learned behavior.

A separate Node comparison checked 203 literal constants in the current
`robot_state.cu` against the extracted configuration. All 29 default joint
angles, action scales, proportional gains, derivative gains and effort limits
match after FP32 conversion; both 29-element policy/MuJoCo index maps match as
integers. This checks source constants only. It does not validate joint-name
binding, filter/history execution, scheduling or the current service settings.
The private `compare-state-constants.cjs` and `state-constants-result.json`
preserve the check. Result SHA256:
`2ce90e2df7f1ea329f86c222ba2395d6537a0a4a0fcd7a9f8569e7fc21b4da98`.

## Evidence and limits

Public source reference: `/home/spark-advantage/codexrook-runtime/upstream/GEAR-SONIC-6733128a3d8a523b1418b06bca3cdf61c8b0987f`. Physical graphs: `/home/spark-advantage/codexrook-runtime/generated/gear-sonic-g1batch8-mode0-20260909T0203Z`; encoder SHA256 `2befe0f7be8e72a33824f04fd39c191287e1e9aab05807012d6ceeb3f688ff40`, decoder `7f4cb9a01bd21bbd0ef96e3c82d2ee1fc015e7a31dfb032d5e844719af3e9de2`.

Private extraction, source, exact commands, stdout/stderr, per-tensor results, and timings are in `C:/rekagent/work/consistent-fighter-20260919-r1/controller-tensor-comparison-r1`; Spark stage is `/home/spark-advantage/rek-training/controller-tensor-comparison-20260921-r1`. Final result `spark-results/result-v2/comparison.jsonl` SHA256: `1b1a804cc253b4810aa53a7c5f138b85c53cecf999cf5df4a3eddabd63777c6a`. Inspector source SHA256: `a6dc9b6f9987ab37670518cd3651b95e25a55ccdd5f9459ca2919c508e4b418b`; binary: `4d495ad91d6a8bcae31cad360e44239b5fb0262f865058d4de7aeb8bc1686889`. Seven final source/output mirror hashes were checked. Payloads and coefficient values remain private.

Current service weights and configuration remain **unknown**. Old-copy/public identity does not establish current-build or server identity, closed-loop trajectory parity, or fighting strength. The concrete next action is to retain the existing controller while correcting independently demonstrated physical measurement mismatches, then perform bounded dynamics measurements. No controller swap is supported as a behavioral intervention by this result.
