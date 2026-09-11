[CmdletBinding()]
param(
    [ValidateRange(1, 20)]
    [int] $TargetCompletedSchedules = 3,

    [ValidateRange(60, 21600)]
    [int] $TimeoutSeconds = 7200,

    [string] $OutputDirectory = 'C:\rekagent\evidence\runtime\rek-controlled-v2'
)

$ErrorActionPreference = 'Stop'
$pipeName = 'rek-ui-bridge-v1'
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$started = [DateTimeOffset]::UtcNow
$runToken = [Guid]::NewGuid().ToString('N')
$transcriptPath = Join-Path $OutputDirectory (
    'rek-controlled-' + $started.ToString('yyyyMMddTHHmmssZ') + '-' + $runToken + '.jsonl')
$summaryPath = [System.IO.Path]::ChangeExtension($transcriptPath, '.summary.json')

$pipe = [System.IO.Pipes.NamedPipeClientStream]::new(
    '.',
    $pipeName,
    [System.IO.Pipes.PipeDirection]::InOut,
    [System.IO.Pipes.PipeOptions]::Asynchronous)
$reader = $null
$writer = $null
$log = $null
$requestNumber = 0
$completedSchedules = 0
$completedRoundKeys = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::Ordinal)
$skippedPairingRoundKeys = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::Ordinal)
$currentScheduleRoundKey = $null
$currentScheduleRunId = $null
$lastState = $null
$lastProgress = [DateTimeOffset]::UtcNow
$deadline = $started.AddSeconds($TimeoutSeconds)
$leaseAcquired = $false
$lastCommandName = $null
$soloRequestedAt = $null
$roundRequestedAt = $null

function Write-TranscriptRecord {
    param([Parameter(Mandatory)] [object] $Record)
    $line = $Record | ConvertTo-Json -Compress -Depth 20
    $script:log.WriteLine($line)
    $script:log.Flush()
}

function New-RequestId {
    param([Parameter(Mandatory)] [string] $Prefix)
    $script:requestNumber++
    return '{0}-{1:D6}' -f $Prefix, $script:requestNumber
}

function Send-BridgeRequest {
    param([Parameter(Mandatory)] [hashtable] $Request)
    Write-TranscriptRecord ([ordered]@{
        event = 'client_request_sent'
        utc = [DateTimeOffset]::UtcNow.ToString('o')
        request = $Request
    })
    $script:writer.WriteLine(($Request | ConvertTo-Json -Compress -Depth 8))
    $script:writer.Flush()
}

function Get-RoundKey {
    param([object] $State)
    $privateAi = $State.private_ai
    if ($null -eq $privateAi -or -not $privateAi.round_active) {
        return $null
    }
    return '{0}:{1}' -f $privateAi.fight_epoch, $privateAi.round_number
}

function Receive-BridgeMessage {
    $remaining = $script:deadline - [DateTimeOffset]::UtcNow
    if ($remaining.TotalMilliseconds -le 0) {
        throw 'control run deadline expired while waiting for bridge data'
    }
    $waitMilliseconds = [Math]::Max(1, [Math]::Min(
        [int]::MaxValue,
        [int][Math]::Ceiling($remaining.TotalMilliseconds)))
    $readTask = $script:reader.ReadLineAsync()
    if (-not $readTask.Wait($waitMilliseconds)) {
        try {
            $script:pipe.Dispose()
        }
        catch {
        }
        throw 'control run deadline expired while waiting for bridge data'
    }
    $line = $readTask.GetAwaiter().GetResult()
    if ($null -eq $line) {
        throw 'control bridge disconnected'
    }
    $script:log.WriteLine($line)
    $script:log.Flush()
    try {
        $message = $line | ConvertFrom-Json
    }
    catch {
        throw 'control bridge emitted malformed JSON'
    }
    if ($message -isnot [pscustomobject] -or
        [string]$message.protocol -ne 'rek.ui_bridge.v1') {
        throw 'control bridge emitted an invalid protocol envelope'
    }

    if ($message.event -eq 'state') {
        $script:lastState = $message
    }
    elseif ($message.event -eq 'schedule_end') {
        if ([string]::IsNullOrWhiteSpace($script:currentScheduleRunId) -or
            [string]$message.schedule_run_id -ne $script:currentScheduleRunId) {
            throw 'completed schedule did not match the active schedule run id'
        }
        if ([string]$message.schedule_id -ne 'rek.private_bot1.baseline.v1' -or
            [string]$message.command_sequence_schema -ne 'rek.client_fixed.command_schedule.v2' -or
            [string]$message.command_sequence_sha256 -ne
                '39aaab9c3156e8f4d114daac4d4328257b81230ec8b8a372ad2739d38754ec0d') {
            throw 'completed schedule did not match the pinned command sequence'
        }
        if ($message.complete -isnot [bool] -or -not $message.complete -or
            $message.reason -ne 'complete' -or
            [int64]$message.schedule_tick -ne 2600 -or
            [int64]$message.client_fixed_substep -ne 26009 -or
            [int64]$message.move_send_completed_count -ne 8 -or
            $message.final_neutral_send_observed -isnot [bool] -or
            -not $message.final_neutral_send_observed -or
            $message.server_acceptance_observed -isnot [bool] -or
            $message.server_acceptance_observed) {
            throw "measured schedule ended unsuccessfully: $($message.reason)"
        }
        if ([string]::IsNullOrWhiteSpace($script:currentScheduleRoundKey)) {
            throw 'completed schedule has no bound fight epoch and round number'
        }
        if (-not $script:completedRoundKeys.Add($script:currentScheduleRoundKey)) {
            throw "duplicate completed schedule in round $($script:currentScheduleRoundKey)"
        }
        $script:completedSchedules++
        $script:lastProgress = [DateTimeOffset]::UtcNow
        Write-Host (
            "Completed controlled schedule {0}/{1} in round {2}." -f
            $script:completedSchedules, $TargetCompletedSchedules, $script:currentScheduleRoundKey)
        $script:currentScheduleRoundKey = $null
        $script:currentScheduleRunId = $null
    }
    return $message
}

function Wait-ForResponse {
    param(
        [Parameter(Mandatory)] [string] $Event,
        [Parameter(Mandatory)] [string] $RequestId
    )
    while ([DateTimeOffset]::UtcNow -lt $script:deadline) {
        $message = Receive-BridgeMessage
        if ($message.event -eq $Event -and $message.request_id -eq $RequestId) {
            return $message
        }
    }
    throw "timeout waiting for $Event response to $RequestId"
}

function Invoke-BridgeCommand {
    param(
        [Parameter(Mandatory)] [string] $Command,
        [switch] $AllowRejected
    )
    $requestId = New-RequestId 'cmd'
    Send-BridgeRequest ([ordered]@{
        type = 'command'
        request_id = $requestId
        command = $Command
    })
    $ack = Wait-ForResponse -Event 'ack' -RequestId $requestId
    if ([string]$ack.command -ne $Command -or
        $ack.server_acceptance_observed -isnot [bool] -or
        $ack.server_acceptance_observed) {
        throw "bridge rejected $Command`: $($ack.reason)"
    }
    if ([string]$ack.status -ne 'accepted') {
        if ($AllowRejected -and [string]$ack.status -eq 'rejected') {
            return $ack
        }
        throw "bridge rejected $Command`: $($ack.reason)"
    }
    $script:lastCommandName = $Command
    $script:lastProgress = [DateTimeOffset]::UtcNow
    return $ack
}

function Request-BridgeState {
    $requestId = New-RequestId 'state'
    Send-BridgeRequest ([ordered]@{
        type = 'get_state'
        request_id = $requestId
    })
    return Wait-ForResponse -Event 'state' -RequestId $requestId
}

try {
    $pipe.Connect(15000)
    $reader = [System.IO.StreamReader]::new($pipe, $utf8, $false, 65536, $true)
    $writer = [System.IO.StreamWriter]::new($pipe, $utf8, 65536, $true)
    $writer.AutoFlush = $true
    $writer.NewLine = "`n"
    $log = [System.IO.StreamWriter]::new($transcriptPath, $false, $utf8, 1MB)
    $log.AutoFlush = $true

    Write-TranscriptRecord ([ordered]@{
        event = 'control_run_start'
        schema = 'rek.controlled_run.v1'
        utc = $started.ToString('o')
        host = [Environment]::MachineName
        target_completed_schedules = $TargetCompletedSchedules
        timeout_seconds = $TimeoutSeconds
        global_input_used = $false
    })

    $hello = Receive-BridgeMessage
    if ($hello.event -ne 'hello' -or $hello.pipe -ne $pipeName -or
        $hello.current_user_only -isnot [bool] -or -not $hello.current_user_only -or
        $hello.local_computer_verified -isnot [bool] -or -not $hello.local_computer_verified -or
        $null -eq $hello.capabilities -or
        $hello.capabilities.exclusive_control_lease_required -isnot [bool] -or
        -not $hello.capabilities.exclusive_control_lease_required -or
        $hello.capabilities.input_available -isnot [bool] -or
        $hello.capabilities.input_available -or
        $hello.capabilities.autonomous_input -isnot [bool] -or
        $hello.capabilities.autonomous_input) {
        throw 'control bridge hello did not prove the expected local lease boundary'
    }

    [void](Invoke-BridgeCommand 'AcquireExclusiveControl')
    $leaseAcquired = $true

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $state = Request-BridgeState
        $privateAi = $state.private_ai
        $lobbyScreen = [string]$state.lobby_screen
        $roundKey = Get-RoundKey $state

        if ($completedSchedules -ge $TargetCompletedSchedules) {
            $roundIsInactive = $null -ne $privateAi -and $privateAi.round_inactive
            $sessionExited = $null -eq $privateAi -and $lobbyScreen -in @('Home', 'FreePlay')
            if ($roundIsInactive -or $sessionExited) {
                break
            }
            Start-Sleep -Milliseconds 1000
            continue
        }

        if ($lobbyScreen -eq 'Login') {
            [void](Invoke-BridgeCommand 'ConfirmLoggedIn')
            Start-Sleep -Milliseconds 750
            continue
        }

        if ($lobbyScreen -eq 'Home') {
            $soloRequestedAt = $null
            $roundRequestedAt = $null
            [void](Invoke-BridgeCommand 'NavigateFreePlay')
            Start-Sleep -Milliseconds 750
            continue
        }

        if ($lobbyScreen -eq 'FreePlay' -and
            ($null -eq $privateAi -or -not $privateAi.proven)) {
            throw (
                'private arena entry is not proven; the public solo route is disabled')
        }

        if ($null -ne $privateAi -and -not $privateAi.proven -and
            $privateAi.network_client_only -and
            $privateAi.context_is_solo -and
            $privateAi.opponent_is_ai -and
            $privateAi.opponent_slot_is_ai -and
            -not $privateAi.human_in_opponent_slot -and
            -not $privateAi.opponent_slot_has_client -and
            -not $privateAi.opponent_human_bit_set -and
            -not $privateAi.exact_sparring_bot_1) {
            throw (
                'unexpected private AI assignment; refusing automatic reset: ' +
                'client_ai_difficulty={0}, sparring_bot_number={1}, phase={2}, ' +
                'round_active={3}, round_number={4}, post_fight_prompt={5}' -f
                $privateAi.client_ai_difficulty,
                $privateAi.sparring_bot_number,
                $privateAi.phase,
                $privateAi.round_active,
                $privateAi.round_number,
                $privateAi.post_fight_prompt)
        }

        if ($null -ne $privateAi -and $privateAi.proven) {
            $soloRequestedAt = $null
            if ($privateAi.round_active) {
                $roundRequestedAt = $null
                if ([string]::IsNullOrWhiteSpace($roundKey)) {
                    throw 'active private Bot 1 round has no stable round key'
                }
                if ([string]::IsNullOrWhiteSpace($currentScheduleRoundKey) -and
                    -not $completedRoundKeys.Contains($roundKey) -and
                    -not $skippedPairingRoundKeys.Contains($roundKey)) {
                    $currentScheduleRoundKey = $roundKey
                    try {
                        $scheduleAck = Invoke-BridgeCommand -Command 'StartMeasuredSchedule' -AllowRejected
                        if ([string]$scheduleAck.status -eq 'rejected' -and
                            [string]$scheduleAck.reason -eq
                                'required_t800_vs_t800_pairing_not_proven:' +
                                'opponent_fighter_robot_id_not_t800') {
                            [void]$skippedPairingRoundKeys.Add($roundKey)
                            Write-Host "Skipped non-T800 opponent in round $roundKey."
                            $currentScheduleRoundKey = $null
                            $currentScheduleRunId = $null
                            continue
                        }
                        if ([string]$scheduleAck.status -ne 'accepted') {
                            throw "bridge rejected StartMeasuredSchedule: $($scheduleAck.reason)"
                        }
                        if ([string]$scheduleAck.schedule_id -ne 'rek.private_bot1.baseline.v1' -or
                            [string]$scheduleAck.command_sequence_schema -ne
                                'rek.client_fixed.command_schedule.v2' -or
                            [string]$scheduleAck.command_sequence_sha256 -ne
                                '39aaab9c3156e8f4d114daac4d4328257b81230ec8b8a372ad2739d38754ec0d' -or
                            [string]::IsNullOrWhiteSpace([string]$scheduleAck.schedule_run_id)) {
                            throw 'schedule start ack did not bind the pinned command sequence and run id'
                        }
                        $currentScheduleRunId = [string]$scheduleAck.schedule_run_id
                    }
                    catch {
                        $currentScheduleRoundKey = $null
                        $currentScheduleRunId = $null
                        throw
                    }
                }
            }
            elseif ([string]::IsNullOrWhiteSpace($currentScheduleRoundKey)) {
                $roundCanBeRequested = $privateAi.post_fight_prompt -or
                                       [string]$privateAi.phase -eq 'Idle'
                if ($roundCanBeRequested -and
                    ($null -eq $roundRequestedAt -or
                     ([DateTimeOffset]::UtcNow - $roundRequestedAt).TotalSeconds -ge 15)) {
                    if ($privateAi.post_fight_prompt -and
                        $privateAi.post_fight_is_winner -eq $false) {
                        [void](Invoke-BridgeCommand 'ExitLostPrivateSession')
                    }
                    else {
                        [void](Invoke-BridgeCommand 'StartRound')
                    }
                    $roundRequestedAt = [DateTimeOffset]::UtcNow
                }
            }
        }

        Start-Sleep -Milliseconds 1000
    }

    if ($completedSchedules -ne $TargetCompletedSchedules) {
        throw "timeout before completing $TargetCompletedSchedules schedules; completed $completedSchedules"
    }

    [void](Invoke-BridgeCommand 'ReleaseExclusiveControl')
    $leaseAcquired = $false
    $ended = [DateTimeOffset]::UtcNow
    Write-TranscriptRecord ([ordered]@{
        event = 'control_run_end'
        schema = 'rek.controlled_run.v1'
        utc = $ended.ToString('o')
        complete = $true
        completed_schedules = $completedSchedules
        elapsed_seconds = [Math]::Round(($ended - $started).TotalSeconds, 6)
        global_input_used = $false
    })
}
catch {
    if ($null -ne $log) {
        Write-TranscriptRecord ([ordered]@{
            event = 'control_run_end'
            schema = 'rek.controlled_run.v1'
            utc = [DateTimeOffset]::UtcNow.ToString('o')
            complete = $false
            completed_schedules = $completedSchedules
            error_type = $_.Exception.GetType().Name
            error = $_.Exception.Message
            global_input_used = $false
        })
    }
    throw
}
finally {
    if ($leaseAcquired -and $null -ne $writer) {
        try {
            $requestId = New-RequestId 'release'
            Send-BridgeRequest ([ordered]@{
                type = 'command'
                request_id = $requestId
                command = 'ReleaseExclusiveControl'
            })
        }
        catch {
        }
    }
    if ($null -ne $writer) {
        try { $writer.Dispose() } catch { }
    }
    if ($null -ne $pipe) {
        try { $pipe.Dispose() } catch { }
    }
    if ($null -ne $reader) {
        try { $reader.Dispose() } catch { }
    }
    if ($null -ne $log) {
        try { $log.Dispose() } catch { }
    }
}

$transcriptHash = (Get-FileHash -LiteralPath $transcriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$summary = [ordered]@{
    schema = 'rek.controlled_run.summary.v1'
    complete = $true
    host = [Environment]::MachineName
    transcript_path = $transcriptPath
    transcript_sha256 = $transcriptHash
    completed_schedules = $completedSchedules
    target_completed_schedules = $TargetCompletedSchedules
    started_utc = $started.ToString('o')
    ended_utc = [DateTimeOffset]::UtcNow.ToString('o')
    global_input_used = $false
}
[System.IO.File]::WriteAllText(
    $summaryPath,
    (($summary | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
    $utf8)
$summary | ConvertTo-Json -Depth 8
