$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\rekagent\work\rek-realtime-20260928-r1'
$helperScript = 'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\native_clone\passive_support\passive.py'
$helperPython = 'C:\Python312\python.exe'
$controlRoot = Join-Path $taskRoot 'viewer-r6-active'
$mirrorRoot = 'R:\pufferlib\rek-evidence\2026-09-28\rek-realtime-r1\viewer-r6'
$remoteRun = '/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r6'
$helperHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $helperScript).Hash.ToLowerInvariant()
if ($helperHash -ne '9d92e0e6516ce46c06cfe98e60e689d36152871e2d3d8b8c336085e067fe7e17') { throw 'Passive helper changed' }
if ((Test-Path -LiteralPath $controlRoot) -or (Test-Path -LiteralPath $mirrorRoot)) { throw 'Fresh control and NAS directories required' }
if (Get-NetTCPConnection -LocalPort 18772 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 18772 is occupied' }
$null = New-Item -ItemType Directory -Path $controlRoot
$tunnelArgs = @($helperScript,'tunnel','--local',$controlRoot,'--remote',$remoteRun,'--port','18772','--until-stop')
$mirrorArgs = @($helperScript,'mirror','--local',$controlRoot,'--remote',$remoteRun,'--port','18772','--nas',$mirrorRoot,'--until-stop')
$tunnelProcess = $null
$mirrorProcess = $null
try {
    $tunnelProcess = Start-Process -FilePath $helperPython -ArgumentList $tunnelArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $controlRoot 'tunnel.stdout.log') -RedirectStandardError (Join-Path $controlRoot 'tunnel.stderr.log')
    $mirrorProcess = Start-Process -FilePath $helperPython -ArgumentList $mirrorArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $controlRoot 'mirror.stdout.log') -RedirectStandardError (Join-Path $controlRoot 'mirror.stderr.log')
    $receipt = [ordered]@{ utc=(Get-Date).ToUniversalTime().ToString('o'); code=$helperScript; code_sha256=$helperHash; tunnel_pid=$tunnelProcess.Id; tunnel_started_utc=$tunnelProcess.StartTime.ToUniversalTime().ToString('o'); tunnel_args=$tunnelArgs; mirror_pid=$mirrorProcess.Id; mirror_started_utc=$mirrorProcess.StartTime.ToUniversalTime().ToString('o'); mirror_args=$mirrorArgs }
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $controlRoot 'LAUNCHED.json') -Encoding utf8
    $receipt | ConvertTo-Json -Depth 6
} catch {
    $launchError = $_
    # Only the fresh helper directory is signalled. Remote viewers are untouched.
    'Helper launch failed; stop owned helpers.' | Set-Content -LiteralPath (Join-Path $controlRoot 'STOP') -Encoding utf8
    $ownedReceipts = @()
    $cleanupDeadline = [DateTime]::UtcNow.AddSeconds(20)
    foreach ($ownedProcess in @($tunnelProcess,$mirrorProcess)) {
        if ($null -eq $ownedProcess) { continue }
        $ownedEntry = [ordered]@{ pid=$ownedProcess.Id; exited=$null }
        try { $ownedEntry.started_utc=$ownedProcess.StartTime.ToUniversalTime().ToString('o') } catch { $ownedEntry.identity_error=$_.Exception.Message }
        try {
            $waitMs = [Math]::Max(0,[int]($cleanupDeadline-[DateTime]::UtcNow).TotalMilliseconds)
            $ownedEntry.exited=$ownedProcess.WaitForExit($waitMs)
        } catch { $ownedEntry.wait_error=$_.Exception.Message }
        $ownedReceipts += $ownedEntry
    }
    [ordered]@{ utc=[DateTime]::UtcNow.ToString('o'); error=$launchError.Exception.Message; owned_helpers=$ownedReceipts; cleanup='Fresh-directory STOP only; no kill escalation' } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $controlRoot 'LAUNCH-FAILED.json') -Encoding utf8
    throw $launchError
}
