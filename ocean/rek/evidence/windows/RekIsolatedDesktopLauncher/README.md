# REK isolated desktop launcher

This Windows-only launcher creates `WinSta0\RekPolicyEval` and uses direct
`CreateProcessW` with `STARTUPINFO.lpDesktop`. It never switches desktops,
corrects focus, sends input, modifies the installed game, or authenticates.
The expected native bridge proof is:

```text
windows_native=1;station=WinSta0;desktop=RekPolicyEval;input_desktop=Default;host=D21
```

The launcher requires host `D21`, station `WinSta0`, and both its own thread
desktop and current input desktop `Default`. The dedicated desktop must not
already exist. A named mutex prevents simultaneous launcher instances.
The desktop handle is retained until the direct child exits, including when
writing evidence fails. The launcher does not terminate existing processes.

## Build and harmless self-test

Build using .NET 8 or a later SDK. Run the resulting apphost `.exe`, not
`dotnet RekIsolatedDesktopLauncher.dll`.

```powershell
dotnet build .\RekIsolatedDesktopLauncher.csproj -c Release -o C:\rekagent\tmp\isolated-desktop-launcher-build
$probe = Start-Process -FilePath C:\rekagent\tmp\isolated-desktop-launcher-build\RekIsolatedDesktopLauncher.exe -ArgumentList '--self-test --output-directory C:\rekagent\tmp\isolated-desktop-probe-NEW' -WindowStyle Hidden -Wait -PassThru
$probe.ExitCode
Get-Content C:\rekagent\tmp\isolated-desktop-probe-NEW\result.json
```

Use a fresh output directory on every run. The self-test runs 16 pure contract
checks, then launches itself with `CREATE_NO_WINDOW` on the dedicated desktop.
The child verifies its actual desktop before creating a harmless non-activating
`STATIC` window there, pumps only that window's messages for 2 seconds, writes
its native proof, destroys its window, and exits. Existing REK processes do not
block the harmless self-test and are never touched by it.

The parent records foreground HWND/PID and input desktop before, during and
after the child. Foreground or desktop changes cause failure; no corrective
focus operation occurs. Sampling every 50 ms does not prove absence of shorter
transients and cannot distinguish human focus changes from child-caused ones.
Artifacts are `launch.json`, `probe.json`, and `result.json`.

## Actual REK launch is a separate, explicit action

Do not use this command until Steam startup behavior, user account and the
native bridge deployment have been reviewed by the operator. The launcher does
not prove account identity or contain a relaunch brokered by an existing Steam
process. It never creates `steam_appid.txt`, transfers tokens, or starts Steam.
Any account sign-in is human-only. A separate native bridge must recheck the
desktop proof dynamically and enforce `moogleod` plus private-AI scope before
semantic actions. A self-test pass does not qualify actual game rendering,
Steam behavior, authentication, private-server admission, or policy parity.

```powershell
# Template only. Supply the independently verified executable SHA-256.
$launcher = Start-Process -FilePath C:\rekagent\tmp\isolated-desktop-launcher-build\RekIsolatedDesktopLauncher.exe -ArgumentList '--launch-rek --exe "C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\REK.exe" --sha256 VERIFIED_64_HEX_DIGITS --output-directory C:\rekagent\tmp\isolated-rek-NEW' -WindowStyle Hidden -PassThru
```

The launcher requires the exact installed executable path and matching supplied
SHA-256, checks that no process named `REK` exists twice before creation, and
holds the hashed executable open without write sharing through child startup.
Its only game arguments are `-screen-fullscreen 0 -screen-width 1280
-screen-height 720`. No argument passthrough exists. Output includes direct
child PID; the launcher remains alive until that child exits. It logs desktop
or foreground changes but does not inject corrective actions or kill processes.
The absence check cannot prevent an unrelated actor from concurrently starting
REK. Close normal REK manually and avoid concurrent starts during this action.

Native references: [STARTUPINFOW](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/ns-processthreadsapi-startupinfow),
[CreateDesktopW](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-createdesktopw),
[CreateProcessW](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw),
[OpenInputDesktop](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-openinputdesktop),
[Steam startup behavior](https://partner.steamgames.com/doc/sdk/api).
