# Native Windows isolated-desktop fallback: read-only investigation

Date: 2026-09-24. No application launched, launch file created, credential store accessed, or Windows UI manipulated during this investigation.

## Finding

Valve documents that a `steam_appid.txt` file makes `SteamAPI_RestartAppIfNecessary` return false, permitting developer/test launches outside the Steam launch broker. This only suppresses that API's restart path. `SteamAPI_Init` still requires a running Steam client, matching OS user/elevation, and a licensed active Steam account. It does not establish REK authentication or private-server admission.

Official primary reference: https://partner.steamgames.com/doc/api/steam_api#SteamAPI_RestartAppIfNecessary (also see `SteamAPI_Init` on that page). The file contains only the App ID; Steam searches the current working directory. Valve says to remove the file before uploading a game depot.

## Verified installed identity and recovered startup

- Installed manifest: `C:\Program Files (x86)\Steam\steamapps\appmanifest_4600200.acf`. Exact fields: App ID `4600200`, name/install directory `REK Alpha Test`, build ID `24969755`.
- Candidate file, if separately authorized: `C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\steam_appid.txt`, contents `4600200`. It was absent when checked. No file was created.
- Current installed `GameAssembly.dll` SHA256, freshly checked: `6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412`.
- Recorded recovery identity: `C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek\evidence\evidence_out\il2cpp_recovery.json:5` records the same build ID and `:14` the same assembly hash. This is static recovery, not a runtime guarantee.
- Recovered `C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp\SteamManager.txt:220` calls `0x1806DB010`; its return is tested immediately. The ISIL at `:444` records that call, `:458` identifies the false-path `SteamAPI.Init`, and `:565` identifies the true-path `Application.Quit`.
- Recovered wrappers `C:\rekagent\work\controller-audit-isil\IsilDump\com.rlabrecque.steamworks.net\Steamworks\SteamAPI.txt:1522` and `NativeMethods.txt:186` identify the shared native restart wrapper. `AppId_t.txt:346` shows its static initializer sets the first static App ID field to zero, which the recovered caller loads.
- Scoped REKApp text/disassembly searches found this restart callsite and no additional `steam://`, `Process.Start`, or `ShellExecute` route. The observed `Application.OpenURL` call is `http://rek.chat/` in `GameMenuController.txt:9795`.

## Existing launcher and containment limit

Launcher source: `C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek\evidence\windows\RekIsolatedDesktopLauncher\Program.cs`. It targets `RekPolicyEval` (`:10`), sets the executable directory as CWD (`:86`), and uses `CreateProcessW` with its isolated desktop startup data (`:110`). It records `steam_relaunch_containment_proven=false` and `account_identity_verified_by_launcher=false` (`:122`). Its adjacent README (`:45`) explicitly warns that an existing Steam process can broker a relaunch outside the launcher's containment.

Existing executable: `C:\rekagent\tmp\isolated-desktop-launcher-20260919-r1\RekIsolatedDesktopLauncher.exe`, previously verified SHA256 `331a3ad2343fa16f10cea40e500aef977df44d4277ca331ef952507d6445c9a1`. Installed REK.exe previously verified SHA256 `5fe6a5c3da371cb7b75a1795eb073b2e69ec718197cf68826bdcd461dcd986c1`.

The documented file supports a bounded developer-launch experiment for the recovered restart check. It does not prove absence of separate native executable, DRM-wrapper, plugin, or external Steam launch behavior. I do not know whether every such route is absent. Complete noninterference remains unproven until a separately authorized, guarded native startup is observed. No native launch is authorized by this report.
