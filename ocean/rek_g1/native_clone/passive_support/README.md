# Passive native clone transport and mirror

Revision 2 adds explicit `--until-stop` for both helpers. That option has no timed expiration; the shared STOP marker, process locks, copy/resume checks and host-key verification remain active. It is mutually exclusive with `--minutes`. Omitting both options preserves the previous 240-minute default, and timed requests still reject values outside `(0,240]` minutes.

To restore the verified existing session, root will use this revision's `passive.py`, the original `passive-support-r1\active` control directory, `--until-stop`, and the exact current remote run. The mirror additionally requires `--resume`, which checks both existing ownership receipts before touching the saved prefixes. Do not copy or recreate the old active directory. This preparation did not launch either helper. No SSH reconnection behavior was changed.

These helpers are staged only. They do not execute commands on Spark, launch or stop the game/viewer, or generate input. The tunnel forwards user HTTP traffic. The mirror reads the chosen isolated run and writes a fresh task-owned NAS subtree.

Two independent Windows processes use Paramiko, the saved `dgx_spark` alias, existing SSH keys/agent and existing known_hosts. Unknown host keys are rejected. The tunnel binds only Windows loopback and forwards to the matching Spark loopback port. The default remains `18771`; `--port 18772` allows a separate tested viewer while preserving the existing session. Ports must be in `1024..65535`. Neither process changes a service or host configuration.

The default source is `/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r2`. Root may supply a later isolated `run-rN` using `--remote`. The destination defaults to `R:\pufferlib\rek-evidence\2026-09-27\rek-native-clone-r1\live-session-r1`. The local control directory defaults to this package's `active` directory, with an independent lock and status for each mode.

After root verifies the final run and approves launching:

```powershell
$code = 'C:\rekagent\work\rek-native-clone-20260927-r1\passive-support-r2\passive.py'
$control = 'C:\rekagent\work\rek-native-clone-20260927-r1\passive-support-r1\active'
$remote = '/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r4'
Start-Process -WindowStyle Hidden -FilePath 'C:\Python312\python.exe' -ArgumentList @($code,'tunnel','--local',$control,'--until-stop')
Start-Process -WindowStyle Hidden -FilePath 'C:\Python312\python.exe' -ArgumentList @($code,'mirror','--local',$control,'--remote',$remote,'--resume','--until-stop')
```

The first mirror run refuses an existing spool or NAS subtree. Resume requires `--resume` and identical local/NAS ownership receipts. Each mode rejects a duplicate process using an OS file lock. If a `STOP` file already exists, no connection is attempted. Create `$control\STOP` to stop both helpers. This does not stop the viewer or game. Timed invocations are bounded to four hours; explicit `--until-stop` has no deadline. A tunnel connection failure stops that helper and records status; it is not silently replaced.

The mirror reads only named configs/logs, recorder health, and numeric PNG frames. Active JSONL/log files are copied incrementally. It checks the last 64 KiB at each destination offset, refuses source shrink or changed prefixes, verifies each new chunk by reading it back, and records that these files are active prefixes. This is an append chain check, not a new whole-source hash on every pass. Immutable configs/frames use complete source/local/NAS SHA256 equality. Incomplete PNGs or changing source files are retried, never published as completed frames. Source originals remain untouched. Health snapshots are retained by source version under `snapshots/`; existing immutable destination bytes may only be accepted when identical.

`mirror-status.json` distinguishes copying, watching, retrying and stopped. It reports source file freshness and the copied recorder heartbeat, plus errors. Copying does not assert that the newest source bytes are already backed up. Local status is written before a NAS operation; an unresponsive SMB call can delay mirror stop/status until the OS returns, while the separate tunnel remains independent. The local `STOP` flag is checked between chunks and requests.

Tests are fake local files/SFTP only, with no remote or viewer connection:

```powershell
& 'C:\Python312\python.exe' -m unittest -v test_passive.py
```

Twelve tests cover append resumption after interruption, prefix-change and truncation rejection, immutable hash/readback and conflict preservation, incomplete PNG rejection, path boundaries, fresh/resumed ownership, STOP-before-connect, exclusive process locking, untimed STOP in both loops, timed default/cap preservation, incompatible option rejection, and independent loopback port selection.
