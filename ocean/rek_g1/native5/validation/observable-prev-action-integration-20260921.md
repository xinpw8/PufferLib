# Previous sampled action: native training and live inference

Date: 2026-09-21. This experiment keeps the reward and physical equations
unchanged. Its hypothesis is that remembering the policy's own sampled choices
helps recurrent control when the observation does not expose accepted controller
commands. No fighting-strength improvement is established by implementation tests.

## Contract

The opt-in schema is `rek.native5.observable_balance_prev_action.v1`.
The [CPU contract and migration](observable-prev-action-cpu-20260921.md)
describe the 33 one-hot cells and availability cell added within existing
metadata padding. All base features and unavailable joint fields retain their
meanings. The history identifies a successful categorical sample. Delivery,
acceptance and execution remain separate facts.

The CUDA trainer wrapper fills history from its bound learner action buffer
after the environment step, on the same CUDA stream. Initial, explicitly reset
and terminal outputs clear history. The overlay does not reset at rollout or
minibatch boundaries. It adds one batched kernel and no CPU physics, tensor
transfer, Python execution or additional device allocation.

Both runtimes continue exporting the unchanged base projection. Startup logs
explicitly identify the required policy-owned augmentation. The wrapper rejects
a feature-mask configuration for this new schema because applying an earlier
runtime mask before the overlay would be ambiguous. Existing schemas are
unchanged.

The live encoder likewise exports measured base features with all 34 history
cells zero. Its manifest identifies worker-owned augmentation. The native worker
inserts its previous successful sample before forward, records the new sample
after successful inference, and clears history with recurrent resets. Malformed
and stale requests do not advance it. Nonzero caller-supplied history is rejected.
The response reports the history actually supplied before feature masking, without
claiming that the request executed.

## Verified live path

Spark stage:
`/home/spark-advantage/rek-training/observable-prev-action-live-20260921-r1`.

The new worker links the unchanged native policy object from the earlier compact
build. The encoder uses no physics or policy runtime. Build and CPU protocol
tests pass for legacy, base observable and previous-action schemas. Encoder
fixtures pass 92 cases per schema: 20,377 assertions for base and 20,571 for
previous-action. All transport history cells remain zero, and existing source
validation, mask gates, point deltas and provenance checks remain exercised.

Actual GB10 inference passes for both sampled and argmax selection:

| Schema | Assertions per selection | Inferences per selection |
| --- | ---: | ---: |
| Base observable | 150 | 6 |
| Previous action | 606 | 39 |

The new-schema tests force all 33 categories and verify the next response's
history. They cover malformed/stale requests, spoofed history cells, explicit
reset, terminal handling and new rounds. These use synthetic observations and
do not launch REK or establish authentic dynamics parity.

| Artifact | SHA256 |
| --- | --- |
| Worker | `46b3d11908347aed544bc2e2e98d54a35aeb6e25cc2473a4ff47312c46a20de2` |
| Unchanged native policy object | `87b59447acfc83bdeb2db0e7698eb47bd203d45370d364993ea8746168ba6580` |
| GPU test script | `3a9bb5040c69a72c3a62ec067a69666f8f3e7db436b74b82328a9c5ee907244d` |
| GPU test runner | `ca87ccf40ffe0f5bbc5fdf5ff0465939ddd0ada7b433d65178c900e3d5c4e9ad` |

These completed build/protocol/encoder/inference records are preserved in
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\observable-prev-action-live-r1\rek-prev-action-live-20260921-r1.tar.gz`.
Spark, Windows and NAS copies agree on SHA256
`724d10c721040086247ee408fce87c9cfb548a7a1e5aa8b27c3f8555ade3121e`.
This archive predates the subsequent migration-equivalence and training tests.

## Checkpoint GPU equivalence

The separately built diagnostic includes the unchanged native policy
implementation and compares the source `ef85a01b...` checkpoint against the
explicitly migrated `ba4339bb...` checkpoint. Eight cases cover BF16/FP32,
sampled/argmax selection, and batch sizes one/four. Each runs 1,300 forwards,
including all 33 history categories, unavailable history, terminals, recurrent
and full resets, graph capture/replay, and 1,100 uninterrupted post-reset steps.

All 884,000 logits/value outputs are bitwise identical. Actual recurrent buffers
are byte-identical, all 26,000 selected actions agree, and native failure bits
remain zero. Maximum absolute logit difference is zero. These synthetic-input
tests establish the measured initial equivalence before retraining; they do not
claim that learned use of the new inputs improves fighting.

The diagnostic ran on Spark in 3.55 seconds, with complete source/commands/output
under the same private stage's `equivalence-source-r1` and `equivalence-r1`.
Result SHA256:
`07317d3ba53d1c921409e2ffbef16dfe040aea0ededaac11721e5990a196c299`.
The earlier migration receipt correctly retains its creation-time statement
that GPU equivalence had not yet been tested.

## Reward retained

`normalized_points_falls_v1` remains
`0.01 * (own awarded point delta - opponent awarded point delta - own became-fallen event)`.
The event penalty occurs once upon confirmed fallen transition. Remaining down,
starting to fall and pose reset do not add repeated penalties. A five-point
award contributes 0.05 through the point delta; there is no separate opponent
fall bonus. Rewards have fixed scaling and safety bounds [-1,1], with saturation
reported. No running-batch reward normalization changes the value of a point.

Compact ordinary-move pretraining has no fall-event producer. Physical
fine-tuning and independent authentic Bot 1 evaluation remain necessary.
Changing observation memory does not repair compact fall dynamics.
