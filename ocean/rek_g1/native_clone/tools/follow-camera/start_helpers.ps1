$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\rekagent\work\rek-follow-camera-20260929-r1'
$helperScript = Join-Path $taskRoot 'deployment\passive.py'
$helperPython = 'C:\Python312\python.exe'
$controlRoot = Join-Path $taskRoot 'viewer-r9-active'
$mirrorRoot = 'R:\pufferlib\rek-evidence\2026-09-29\rek-follow-camera-r1\viewer-r9'
$remoteRun = '/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r9'
$helperHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $helperScript).Hash.ToLowerInvariant()
if ($helperHash -ne 'c21796cf490a4f0c172a7454d65d6974e3ebbb95eb18bc8c4b6948f80cfec52a') { throw 'Passive helper changed' }
if ((Test-Path -LiteralPath $controlRoot) -or (Test-Path -LiteralPath $mirrorRoot)) { throw 'Fresh control and NAS directories required' }
if (Get-NetTCPConnection -LocalPort 18775 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 18775 is occupied' }
# Root fetches the closed startup receipt before invoking this helper stage.
$started = Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'deployment\STARTED.json') | ConvertFrom-Json
if (($started.url -ne 'http://127.0.0.1:18775/') -or ($started.paused -ne $true) -or ($started.tick -ne 0)) { throw 'Fresh paused startup receipt required' }
if ($started.process.source_manifest_sha256 -ne 'e8e52f788603b95deb0935fe9796a05ec8b13b5d780ebee2462ee35d37cc7ad7') { throw 'App startup identity differs' }
if ($started.process.binary_sha256 -ne 'dba48c455123a4fa7d31c34034102b3756dfd469c26963823eb8711c92b5661f') { throw 'Native startup identity differs' }
$null = New-Item -ItemType Directory -Path $controlRoot
$tunnelArgs = @($helperScript,'tunnel','--local',$controlRoot,'--remote',$remoteRun,'--port','18775','--until-stop')
$mirrorArgs = @($helperScript,'mirror','--local',$controlRoot,'--remote',$remoteRun,'--port','18775','--nas',$mirrorRoot,'--until-stop')
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
