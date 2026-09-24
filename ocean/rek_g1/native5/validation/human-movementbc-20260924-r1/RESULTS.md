# Five-epoch movement BC: negative result

Completed native training in **45.4527 s**, exit 0, 165 updates. Final checkpoint `df2ffa1ef1b2dbf97c1d62f8371e6473e33f557f6d04d125974f0646fecda67f` was **not promoted or deployed**. This package records the completed five-epoch experiment. A separate conditional movement-25 preparation existed elsewhere and was never run; it contributes no result here.

## Supervised fit

| Measurement | Training | Development |
| --- | ---: | ---: |
| Movement labels | 1,355 | 1,625 |
| Initial conditional CE | 7.56477 | 6.09681 |
| Final conditional CE | 4.10310 | 4.78090 |
| Final accuracy | 28.3395% | 7.4462% |
| Forward action 2 recall | 0/499 | 0/932 |
| Positive-yaw action 6 recall | 73.3333% | 4.8469% |
| Negative-yaw action 7 recall | 68.3962% | 95.3271% |

Both final cross-entropies exceed uniform 14-class CE, ln(14) = 2.63906. Forward failure even on training labels establishes incomplete fitting. No epoch was selected using development metrics: epoch 5 was fixed prospectively. The native field name `heldout` means the already examined development split here, not an untouched final test.

## Fixed-stream drift

The offline diagnostic reused 787 full-223/all-ones requests from `humanbc-s1101`, including its final rejected action. Baseline reproduced 787/787 actual samples. That source ended under the action watchdog at a **5:19 partial score**, 4.4455 s remaining, with the round still active. It was not a completed round or a counterfactual rollout.

Candidate changed 192/787 sampled actions; mean/max legal KL was 0.0316934/0.364246. Sampled movement changed 735 to 724; sampled attacks 6 to 10. Category 17 stayed zero. Neither policy selected forward action 2 as the conditional movement argmax. Legal movement mass fell from 0.928418 to 0.921840; unmasked movement mass fell from 0.621852 to 0.547105. Shared-network training changed attack mass despite a movement-only conditional loss. None of these changes establish better fighting.

## Interpretation limits

The two human rounds are from one session. Labels represent temporal occupancy of exact outgoing commands, not independent starts or verified physical keys. Training retained 20 ms chronology and partial observations; live inputs were full-featured with variable cadence. Commands 2/6/7/9 retain tuples `(1,0,0)`, `(0,0,1)`, `(0,0,-1)`, `(1,0,-1)` in forward/strafe/yaw order. Archived identity/sign review found no own/opponent swap; coverage differed between rounds. Coverage concerns do not explain away zero training forward recall.

Execution, every epoch's scalar and nonzero-label per-action metrics, drift report and source/archive hashes accompany this summary. Private observations, datasets and checkpoint bytes remain outside the repository.
