# Recorded root motion during same-request busy intervals

Existing r21-r23 observations contain substantial planar root motion during
same-request busy windows, including windows whose sampled root separation
stays outside the compact runtime's body-overlap threshold. This establishes
a missing observed-motion consequence in the compact locked-action model.
It does not identify that motion as an executed attack's causal displacement,
or determine how much comes from held yaw, animation, contact, or inertia.

No game interaction, new capture, GPU work, fitting, or runtime change was
performed for this measurement.

## Definition and result

Inputs are the finalized `encoder.stdin.jsonl` and `encoder.stdout.jsonl`
under `C:\rekagent\work\consistent-fighter-20260919-r1\live-round_outcome_v1-rN\trial`
for N = 21, 22, 23. Ready encoder outputs are joined to their input by
observation sequence and requested-move QPC identity. Busy means the saved
`provenance.projected_busy`, specifically `dispatched_request_v4_duration`.
It is a request-duration projection, not measured authoritative playback.

A window is a contiguous run of busy, nonterminal observations with the same
`requested_move_qpc_ticks`; a QPC observation gap greater than 0.25 s breaks it.
At least two samples and at least 0.30 s are required for the filtered counts.
The planar position is observation columns 0 and 1, converted Unity x/z.
Net displacement is `hypot(x_last-x_first, y_last-y_first)`, not path length.
Net yaw is the sum of wrapped consecutive differences of
`2*atan2(obs[175],obs[172])`, reported in degrees.

The default compact body radius is 0.22, so its root-overlap correction operates
below separation 0.44. Its floor half-extent is approximately 2.30, giving a
root confinement bound of approximately 2.08. The upright subset requires,
at every saved sample: no observed fallen/falling flag, tilt below 30 degrees,
height above 0.50, and absolute actor x/y below 2.08. These position thresholds
are captured numeric units under the existing 1:1 projection assumption;
metric calibration is not asserted.

| Round | Same-request busy windows, before duration filtering | Separation >0.44, duration >=0.30 s | Of those, net displacement >0.10 | Upright/in-bounds subset | Upright net displacement >0.10 | Stricter separation >0.60: displaced / eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r21 | 93 | 85 | 29 | 79 | 24 | 11 / 52 |
| r22 | 121 | 99 | 23 | 99 | 23 | 12 / 65 |
| r23 | 102 | 82 | 23 | 77 | 22 | 14 / 46 |

Thus 69/255 upright, in-bounds windows exceed 0.10 net displacement while
sampled separation remains above 0.44. A larger separation margin retains
37/163 above 0.10. Median net displacements in the first upright subset are
0.0733, 0.0600, and 0.0700 respectively.

Concrete upright, in-bounds examples, all with separation above 0.60:

| Round / requested native move | Source sequences | Request QPC ticks | Duration, s | Minimum separation | Net planar displacement | Net rendered yaw, degrees |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r21 / 10 | 1519-1645 | 5520890754791 | 2.6316 | 0.7020 | 0.5520 | +25.35 |
| r22 / 10 | 344-476 | 5522748447473 | 2.6415 | 0.6672 | 1.0187 | +12.90 |
| r22 / 3 | 3026-3070 | 5523298170608 | 0.8759 | 0.7799 | 0.8267 | +134.75 |
| r23 / 3 | 4788-4832 | 5526175740899 | 0.8679 | 0.6585 | 0.7899 | +34.72 |

The largest observation gap in these examples is 0.0499 s. Sampled separation
does not prove continuous separation between samples, and separation above
0.44 does not exclude limb contact in the authentic client.

## Interpretation and limits

In `fast_runtime.cu`, `advance_fighter` returns from the attack branch before
root translation and logical yaw integration. `advance_arena` can subsequently
change planar roots through body-overlap correction or confinement. Neither
mechanism represents motion from attack clips, physical impulses, or balance.
The reported examples remain outside those sampled root constraints. The
compact asset bake uses clip root height but omits clip root x/y displacement.

Rotation needs a separate interpretation: compact rendered root rotation is
logical yaw composed with clip-root rotation. A changing observed pelvis yaw
therefore does not by itself prove changing logical/controller yaw. The
planar discrepancy is the stronger finding here.

Current recordings do not cleanly separate sustained held-yaw response from
attack-driven displacement. No eligible complete busy window holds zero yaw
intent throughout. Splitting into constant-intent, upright, separated
subintervals of at least 0.30 s gives only six neutral and two negative-yaw
intervals across all three rounds, and no positive-yaw interval. The two
negative-yaw intervals have net rendered yaw changes of -26.57 and +8.04
degrees. Neutral intervals last only 0.32-0.41 s, shorter than the configured
0.5 s yaw ramp. Prior command history, asynchronous response, clip rotation,
contact and inertia remain confounded. Dispatch is not execution evidence.

The separately reported 10/93, 53/121, and 47/102 counts concern absolute
opponent bearing above 90 degrees at attack request. They are not rotation
or displacement counts and are not compared to the net-yaw measurements here.

The next useful physical comparison is a matched native-physical replay of
the observed move-3/right-hook and move-10 contexts, with the same initial
pose, gap, attack phase, retained-yaw history and pelvis projection. Unknown
server contacts and initial velocities must remain explicit limitations.
These mixed observational windows do not justify fitting an invented slider
residual or a generic attack-displacement correction.

## Reproduction and source binding

This read-only PowerShell command reproduces the upright/in-bounds counts and
the stricter separation check directly from saved encoder outputs:

```powershell
@'
import fs from 'node:fs';
import rl from 'node:readline';
const root='C:/rekagent/work/consistent-fighter-20260919-r1/';
for (const round of [21,22,23]) {
  const groups=[]; let current=[];
  const file=root+`live-round_outcome_v1-r${round}/trial/encoder.stdout.jsonl`;
  for await (const line of rl.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity})) {
    const e=JSON.parse(line); if(!e.ready) continue;
    const p=e.provenance, o=e.worker_request.observation;
    const t=p.source_qpc_ticks/p.source_qpc_frequency_hz;
    const last=current.at(-1);
    if(last && (!p.projected_busy || e.worker_request.terminal ||
       last.request!==p.requested_move_qpc_ticks || t-last.t>.25)) {
      if(current.length>1) groups.push(current); current=[];
    }
    if(p.projected_busy && !e.worker_request.terminal) {
      const s=p.observed_source_values_excluded_from_structural_features.find(v=>v.slot===o[184]);
      current.push({t,request:p.requested_move_qpc_ticks,x:o[0],y:o[1],height:o[2],gap:o[86],s});
    }
  }
  if(current.length>1) groups.push(current);
  for(const gap of [.44,.60]) {
    const eligible=groups.filter(a=>a.at(-1).t-a[0].t>=.30 && a.every(v=>
      v.gap>gap && !v.s.fallen && !v.s.falling && v.s.tilt_degrees<30 &&
      v.height>.50 && Math.abs(v.x)<2.08 && Math.abs(v.y)<2.08));
    const displaced=eligible.filter(a=>Math.hypot(a.at(-1).x-a[0].x,a.at(-1).y-a[0].y)>.10);
    console.log(JSON.stringify({round,gap,eligible:eligible.length,displaced:displaced.length}));
  }
}
'@ | node --input-type=module -
```

SHA-256 bindings for the original finalized inputs:

| Round | Encoder stdin | Encoder stdout |
| --- | --- | --- |
| r21 | `6ccbd6eccfa18bec02c95517c46dbdd31b8a24dec031a3876221d3b97c256ddd` | `c21a81a1b0e266775771e64e18df52454c46281a6bcc06daf34cd295164815ba` |
| r22 | `17cd3f415777a3d6883748462834c862270beb04c2ce0383a73e0634ef030fc5` | `94a635ef79bbf20ca73a769298532e0482874c624f05d65a1400b89caf56baad` |
| r23 | `e09ee1a547c1580a1081f0d453533d01431cff51b9f0d4cc5eb5b9b0ea4409ce` | `d86927c939045368b13ebc79688a9b9ef98ac514a1640b5c48df8b71cd02ff5d` |

All three encoder manifests bind model SHA-256
`6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa`.
