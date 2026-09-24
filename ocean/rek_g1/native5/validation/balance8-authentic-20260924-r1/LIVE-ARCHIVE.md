# First three completed live rounds: preserved evidence

Cutoff: completed `balance8-s903-retry3`, 2026-09-24 07:37:02 UTC. Candidate result at this cutoff: **1 win, 2 losses; 27:39 points**. The campaign continued after this snapshot. This is not its final result.

Archive on the physical evidence server:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-24\balance8-live-first3-20260924T073702Z\evidence.tar.gz`

- Size: 114,393,738 bytes.
- SHA256: `5cda55bec2c98ddf80b400b2ca45c0b36f987ff61b71a6e77857d933e9c97e80`.
- SSH tar exit: 0. NAS readback SHA256 matched the transfer stream.
- Independent tar listing: exit 0, 279 entries.
- Includes all ten closed attempts for seeds 901, 902, and 903, their exact configs, fixed plan, driver, recorder, and controller. Seven attempts were outside the complete 120-second criterion; their artifacts remain included.
- Native training source, replay, and checkpoint archive is documented separately in `ARCHIVE.md`. No source files were removed.

Standalone authentic 13:3 win video in the same NAS directory: `round-s903-13-3.mp4`, 10,884,271 bytes. SHA256 `2b1d267d5feb41aa3a0c89de4b32fe3644b0f6c65b91429772d2c3e1eafbeb9b`. Native FFmpeg decode passed; local and NAS copies matched. The preceding losses were 5:13 and 9:23 and are preserved in the archive.

Transfer receipt: `C:\rekagent\work\authentic-ppo-live-20260924-r1\balance8-first3-archive-receipt.json`, also `receipt.json` next to the archive. This report contains no proprietary game binaries or credentials.
