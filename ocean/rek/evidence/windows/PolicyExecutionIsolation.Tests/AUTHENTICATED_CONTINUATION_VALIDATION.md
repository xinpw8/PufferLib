# Native already-authenticated continuation, r2

Date: 2026-09-19. Built and tested without deployment, restart, game commands,
authentication calls, desktop switching, or global input.

## Recovered boundary

- `LoginScreenController.txt:6211` contains `OnLetsGoClicked`: it invokes
  `OnLetsGoPressed` only.
- `LobbyShellController.txt:3280-3295` wires that event to `HandleLetsGo`.
- `LobbyShellController.txt:8600` contains `HandleLetsGo`: it calls
  `ShowScreen(Home=2)` only.
- `LoginScreenController.txt:4470` contains `ShowLoggedInState`: it sets the
  `isLoggedIn` flag, shows the logged-in pane, hides logged-out/attempting panes,
  and supplies the fighter-name label. The native Intro completed independently
  in the parent's passive startup check; no Intro bypass was added.

Recovered files are under
`C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp`.
Current interop public fields/method were inspected before implementation.

## Implemented guard

Native `ConfirmLoggedIn` has a separate one-shot pre-lease route. All ordinary
native leased commands and the initial Home account pin keep their r1 guards.
Spark follows its existing command path. Relay bootstrap is unchanged.

The continuation requires exact native D21/WinSta0/RekPolicyEval/Default proof,
no connected game session, no lease/pin/control mode or earlier attempt, genuine
Login screen, `isLoggedIn=true`, visibly shown logged-in pane, hidden logged-out
and attempting panes, exact `moogleod` in both fighter-name label and GameContext,
and a shown/enabled Let's go button. Visible checks include attached panel,
positive finite bounds, and ancestor display/visibility/opacity.

The native surface and UI facts are reread immediately before the sole callback.
`OnLetsGoClicked` is the only game-mutating method called. No auth client method,
credential getter, keyboard/mouse/gamepad event, or direct screen mutation is used.
Success requires the genuine Home screen, Home display `moogleod`, and unchanged
GameContext identity. A callback/postcondition failure cannot be retried by this
process. The continuation does not grant or pin a lease.

Top-level `state.login` reports the token-free readiness facts and rejection
reason. It does not claim server-side token validity or expose email/token data.

## Tests and artifacts

- Isolation/readiness: 5,930 passing checks, including all 512 Boolean readiness
  combinations, name/context mismatches, every pre-callback surface boundary,
  UI changes between reads, callback order, exceptions, one-shot behavior, and
  failed Home postconditions.
- Existing relay: 6,447 passing checks.
- Existing protocol: 441 cases plus current-user random pipe roundtrip passed.
- Real passive API query still rejects execution on Default desktop.
- Builds: zero warnings/errors. `git diff --check` passed.

`C:\rekagent\tmp\windows-isolation-20260919-r2\bridge\RekUiBridgeAgent.dll`
SHA256: `f868b55ffa6639df32e6c98d42775676a2c641c61047085eebf2f88ffe07b9d5`.

R2 isolation test assembly SHA256:
`479f5ffdaea60d329daa7156f59e6c57e7e5df93fc45eaccf579e629425804b6`.
The final test fixtures use a reserved example email domain. The parent reran
all above suites and 25 live-transfer driver tests successfully; the bridge
binary was unchanged by that fixture-only update.

The relay binary is unchanged from r1:
`ce4e034a07587dbd618e98615cf2ba603bc9149ecd994c683d37a29034aadcbf`.
Plugin version remains 0.4.9 with exact new binary hash required by the relay.
GameAssembly, metadata, and assets hashes remain the pinned r1 values.

Runtime readiness on the actual Login UI and the Home postcondition are not
validated by this build/test subtask. The parent owns any user-approved startup
and one-shot continuation; it must stop if readiness is absent.
