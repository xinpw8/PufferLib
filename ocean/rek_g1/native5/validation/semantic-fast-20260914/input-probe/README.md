# Input and deterministic contact probe

Executed on `spark-4ae3` against the unchanged `build-v1` fast runtime objects.
Compilation and execution exited 0. Source, object and executable hashes,
commands, stdout, stderr and process timing are included. No binary or private
model is published.

Passed in a four-arena run. Behavioral assertions inspect arena 0; final
sticky-status checks cover all four arenas:

- Held Q establishes yaw, an attack interrupts it, and Q resumes afterward.
- Desired E replaces Q during an attack; neutral releases it.
- Additional attacks and translation entered during an attack are discarded.
- The 157-tick attack opens its mask exactly at completion.
- Translation blocks attack entry until it settles; the blocked attack does
  not queue automatically.

Two identical 6,400-tick scripted contact replays produced bitwise-identical
final qpos and matching outcomes. Arena 0 completed nine rounds and scored 220
points across those completed rounds. The periodic diagnostic snapshots saw
at most 21 points in a round and two opponent knockdowns before subsequent
reset/round transitions. These maxima are sampled every 50 ticks, not a claim
that no third knockdown occurred between samples.

Contact and knockdown rules here belong to the explicitly approximate
`semantic_cuda_v1` candidate. This verifies its reproducibility and input
semantics; it does not establish REK gameplay parity or policy strength.
All state advancement occurred in CUDA. Host code performed initialization,
diagnostic snapshots and assertions; no Python or CPU physics ran.
