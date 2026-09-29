# Daniel's saved REK controls

Use the corrected viewer at http://127.0.0.1:18773/. Click Play or the arena to give the arena keyboard focus. These bindings match the current Windows REK control values verified on 2026-09-29.

| Keys | Action |
|---|---|
| W / S | Forward / backward |
| A / D | Strafe |
| Q / E | Turn |
| I | Left hook |
| K | Left jab |
| O | Right hook |
| L | Right jab |
| YY | Left side kick |
| HH | Left front kick |
| UU | Right side kick |
| JJ | Right knee |
| Space + J | Double uppercut |
| Space + L | Left jab, right uppercut |
| Space + Y | Six-punch combo |
| Space + U | Run and punch |
| ; | Left-right jab |
| ' | Left-right hook |
| Space + K | Left hook, right jab |
| Space + H | Double hook |
| Space + I | Butt-smack emote |
| Escape | Pause the viewer |

For YY, HH, UU and JJ, press, release, then press the same key again within 300 ms. Holding a key or operating-system key repeat does not count as a double tap. For a Space chord, hold Space while pressing the other key. Movement and attacks can be combined.

The native controller may reject a requested attack while another move is active, while recovering, or while the round is inactive. The command status and recordings distinguish rejection from acceptance. The correction maps all 17 named commands to their original native move indices; it does not bypass those native gates.

Port 18772 preserves the previous session and incorrect move translation. Its emitted commands and outcomes remain recorded as observed. [Mapping correction and tests](validation/controls-fix-20260929-r1/RESULTS.md) document the defect.
