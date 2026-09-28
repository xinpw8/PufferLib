$ErrorActionPreference='Stop'
$taskRoot='C:\rekagent\work\rek-native-clone-20260927-r1'
$taskScript=Join-Path $taskRoot 'passive-support-r2\passive.py'
$taskControl=Join-Path $taskRoot 'passive-support-r1\active'
$taskReceiptDir=Join-Path $taskRoot 'reconnection-r1'
$taskRemote='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r4'
if(Test-Path -LiteralPath $taskReceiptDir){throw 'Fresh reconnection receipt directory required'}
if(!(Test-Path -LiteralPath $taskScript)){throw 'Verified replacement helper required'}
if(Test-Path -LiteralPath (Join-Path $taskControl 'STOP')){throw 'STOP remains authoritative'}
if(Get-NetTCPConnection -LocalPort 18771 -State Listen -ErrorAction SilentlyContinue){throw 'Preserving existing listener'}
New-Item -ItemType Directory -Path $taskReceiptDir | Out-Null
foreach($taskName in @('STARTED.json','tunnel-status.json','mirror-status.json','tunnel.stdout.log','tunnel.stderr.log','mirror.stdout.log','mirror.stderr.log')){
    Copy-Item -LiteralPath (Join-Path $taskControl $taskName) -Destination (Join-Path $taskReceiptDir "prior-$taskName")
}
$taskProcesses=@()
foreach($taskMode in @('tunnel','mirror')){
    $taskArgs=@($taskScript,$taskMode,'--local',$taskControl,'--remote',$taskRemote,'--until-stop')
    if($taskMode -eq 'mirror'){$taskArgs += '--resume'}
    $taskProcess=Start-Process -WindowStyle Hidden -FilePath 'C:\Python312\python.exe' -ArgumentList $taskArgs -PassThru -RedirectStandardOutput (Join-Path $taskReceiptDir "$taskMode.stdout.log") -RedirectStandardError (Join-Path $taskReceiptDir "$taskMode.stderr.log")
    $taskProcesses += [ordered]@{mode=$taskMode;pid=$taskProcess.Id;startedUtc=$taskProcess.StartTime.ToUniversalTime().ToString('o');args=$taskArgs}
}
$taskReceipt=[ordered]@{utc=(Get-Date).ToUniversalTime().ToString('o');scriptSha256=(Get-FileHash -LiteralPath $taskScript -Algorithm SHA256).Hash.ToLower();untilStopped=$true;processes=$taskProcesses;reason='Original four-hour tunnel/mirror lifetime expired; Spark service remained healthy'}
$taskReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskReceiptDir 'STARTED.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 6
