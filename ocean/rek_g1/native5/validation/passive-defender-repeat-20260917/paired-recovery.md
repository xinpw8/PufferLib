# Paired neutral-defender recovery runs R11 and R12

These two completed collections add 12 received native hit packets: seven non-kick and five kick. Their 17 score packets include five separate +5 awards. This report is separate from the eight-capture recovery aggregate. It does not establish a known AI action-position combination that reliably produces contact.

[paired-recovery.json](paired-recovery.json) records full provenance, hashes for 71 run artifacts and both native inputs, decoded packet values, receipt clocks, pose availability, command counts and validation results. Both runs observed Sparring Bot 1, difficulty 0, in the guarded isolated private-AI scope. No policy, encoder or checkpoint was used.

## Collection and neutral-control evidence

| Collection | Entry and attachment | Relay sources, all neutral | Action-1 requests / applied / rejected | Native hits, non-kick / kick | Native scores / separate +5 | Final source points, slots 0 / 1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R11 | `safe_start`, native ready requested; first source at 119.6666 s remaining | 5,649 | 5,648 / 5,646 / 2 | 5 / 1 | 7 / 1 | 5 / 7 |
| R12 | `follow_on`, attached to existing round 2 at 110.497284 s remaining | 5,297 | 5,296 / 5,295 / 1 | 2 / 4 | 10 / 4 | 20 / 10 |

Both stop on the terminal source, report `completed_round`, exit with code 0 and verify stream-stop plus lease-release cleanup. All 10,944 recorded action requests are action 1. R11 records one stale-observation rejection and one terminal `policy_stream_not_owned` rejection; R12 records one terminal `policy_stream_not_owned` rejection. Neither records an unmatched acknowledgement or an in-flight action at stop.

The native recordings cover approximately 120 s each. R11 has 6,002/6,002 compact samples with velocity `[0,0,0]`, no pending move and `punching=false`, plus 6,213/6,213 zero outbound velocity requests. R12 has 6,001/6,001 neutral compact samples and 6,350/6,350 zero velocity requests. Each capture footer contains only `SendVelocityCommand`, finalizes on `scope_exit:round_not_active`, reports zero capture errors and matches the observed packet counts. Neutral control leaves physical motion and opponent AI active.

Independent relay auditing confirms all active-source desired actions are 1 and measured commands are zero. R12's audit observes nine score-counter updates after attachment: local +20 and opponent +8. The full native capture contains ten score packets because one opponent +2 preceded attachment.

## R12's 20 points and the attachment boundary

R12's local 20 points are exactly four decoded slot-0 +5 awards, with `new_hit_count` values 5, 10, 15 and 20. There are no slot-0 +1 or +2 score packets. Opponent 10 points consist of four +2 awards and two +1 awards. None of the four +5 packets has a same-frame hit receipt. The packet data does not identify the +5 causes; this report makes no knockout, fall, push or strike attribution.

The native R12 capture starts at 08:31:40.5010283 UTC. The first relay source is 08:31:49.9234005 UTC, 9.422334 s later by the two measured QPC values. Its round counter is already `[0,2]`.

One native kick receipt and its noncausal same-frame +2 score precede that first relay frame:

- Packet Unity frame: 9973; Unity time: 185.37119095804988 s.
- First relay frame: 10073; Unity time: 187.07363825821994 s.
- The packet's Unity time is 1.70244730017006 s before the first relay sample. This is a client Unity-clock difference, not server impact latency.
- The paired join reports `partial_bracket`, with no prior or same-frame relay pose. Its only next pose is the later first relay observation. Event-position context from the relay is unavailable, and no pose is projected backwards.

All five later R12 hits and all six R11 hits have prior, next and same-frame relay samples with status `bracketed`. The raw native files also contain root samples, but those have not been substituted for missing relay poses in these paired joins. R11's first relay source is 0.4691074 s after native capture start; it also is not a first-tick whole-round relay capture, although no hit or score receipt precedes its first source.

Every hit has one same-frame score association, explicitly noncausal. Decoded `is_kick=0` accompanies +1 and `is_kick=1` accompanies +2 in these received records. Hit packets lack fighter identity and a shared authoritative contact identifier. Known AI clip ID, strike side, target body zone, rejected attempt and authoritative collision time remain unavailable. The hit channel is unreliable, so a missing receipt leaves contact unknown.

## Provenance and media

All local run paths are below `C:\rekagent\work\passive-defender-20260917-r1\`. Each `round-r11` / `round-r12` directory contains `trial`, `audit-r1`, `native-hit-join-r1` and `media`. Native files are in the common `native` directory.

| Input | SHA256 |
| --- | --- |
| R11 native `rek-private-ai-root-motion-20260917T082930.7952252Z-pid1216-4c2c69d64b6a4c649011c6c01f0aa180.jsonl` | `aaeec92119dcc193fb09be382d24db78cce5a0182dbd42543b117d48326d6b54` |
| R12 native `rek-private-ai-root-motion-20260917T083140.5010283Z-pid1216-2781a646178f41ca9d3e44b146b3d24d.jsonl` | `b271470c0516cd44fc5b5f6da7598bceea6f32d884ee5c7cb41e9f95aab85a0c` |
| R11 `trial/g1_policy_state.jsonl` | `4ed8bba870cbeca13d6421c874e2234dcf829d5d14dfefd9eaa60e36f44fbb1a` |
| R12 `trial/g1_policy_state.jsonl` | `bc5238aefeef583823abf962310701bb604820af6c982a2464fff196726e7fc0` |

The recorder remains version 0.7.2, SHA256 `a19f619c83eeecf9c6ccf79adf339be1f7f1cca8e3cd622f80616f268aaffa95`. Both recorded relay commands explicitly pin UI bridge SHA256 `6e15abbde6fdcd07f5ac71c632fd67cc2553d0a5cc35de34c9c81603c9880276`. These runs are distinct from the earlier bridge configuration. The staged runner/helper identities are retained in the JSON.

| MP4 | Bytes | SHA256 |
| --- | ---: | --- |
| R11 `media/authentic-rek-passive-defender.mp4` | 11,033,849 | `6a375602b93d5f08c7766c4ff0442760b90d8b062bed81db65aaf52f90c56268` |
| R12 `media/authentic-rek-passive-defender.mp4` | 10,272,387 | `cc99bb05572e8efb6ebd2dc45d4154b345335d14c8a50748e6bcc9088e80a727` |

Both MP4 hashes and sizes were independently checked against their capture manifests. Both are below 20,000,000 bytes. The saved manifests record successful full decode and `validated_mp4`. No additional visual interpretation is asserted by this report.

R12's finalized folder and native file were copied from Spark by exact path, without overwriting an existing local run. Remote SHA256 values match the local native capture, filtered source, relay stdout and MP4. The offline audit and join both returned code 0. Independent streaming checks rehashed both native files, compared all 12 raw hits and 17 raw scores with their joined copies, checked both source streams and all 10,944 requested actions, and hashed 71 run artifacts. No mismatch was found. Exact offline commands and tool hashes are in the JSON. No game input, deployment or commit was performed for this report.
