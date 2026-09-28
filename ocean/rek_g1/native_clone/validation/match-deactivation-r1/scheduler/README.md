# Native match input deactivation

Adds a physical-controller Deactivate operation and match-only input activity mask. Existing pre-step APIs and state layout remain unchanged.

The deactivation edge calls CancelAction unconditionally, then ordinary PlayIdle. It zeroes effective velocity and clears locomotionActive, stopBraking, and hasMomentum. It preserves transitionSettling, passive route/last/brake fields, heading, dispatch mode, and categorical adapter ownership. The expired tick-local semantic output is cleared as adapter bookkeeping.

Inactive match rows skip all command, handoff, and locomotion dispatch. Unsuspended rows still sample the current motion reference. Direct feedback reports clone-specific INPUT_INACTIVE (4), with move rejection only when a move was requested. Runtime owns the phase mask and edge timing.

CPU validation: 143 new assertions, 216 ordered reset assertions, and 21,839 existing scheduler assertions passed; the latter includes 1,200 disabled-equivalence ticks. Tests execute the actual scheduler kernel bodies on CPU with synthetic motion clips. They do not establish physics or official-Unity end-to-end parity. An initial fixture-only compile error (two mixed-type auto declarations) is preserved under cpu-run-r1; production source was unchanged for the corrected cpu-run-r2.

Recovered authority: RobotInputController.txt, Deactivate, original CancelAction at line 11179, IsVisualOnly branch at 11223, PlayIdle at 11238 and three cleared flags at 11241-11243. Source path and full hash are in AUTHORITY.json.
