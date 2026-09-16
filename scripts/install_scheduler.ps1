<#
    Register the 24/7 jobs with Windows Task Scheduler.

        powershell -ExecutionPolicy Bypass -File scripts\install_scheduler.ps1
        powershell -ExecutionPolicy Bypass -File scripts\install_scheduler.ps1 -Remove

    Creates two tasks:

      Loonie-Evolve   continuous strategy search, restarted at boot and if it
                      dies. Checkpoints every generation, so a restart costs
                      at most one generation.
      Loonie-Trade    one paper rebalance per weekday, 15:45 America/New_York
                      (15 minutes before the close, so market-on-close orders
                      still make the cut).

    Both run PAPER. Neither can arm live trading: that needs config.yaml,
    ALPACA_MODE, and a CLI flag this script does not pass.
#>

param(
    [switch]$Remove,
    [string]$TradeTime = "15:45",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    throw "python not found on PATH; pass -PythonExe C:\path\to\python.exe"
}

$tasks = @("Loonie-Evolve", "Loonie-Trade")

if ($Remove) {
    foreach ($t in $tasks) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $t -Confirm:$false
            Write-Host "removed $t"
        }
    }
    exit 0
}

Write-Host "project : $root"
Write-Host "python  : $PythonExe"

# ---- Evolve: continuous, restart on failure or boot ----------------------
$evolveAction = New-ScheduledTaskAction -Execute $PythonExe `
    -Argument "scripts\run_evolve.py --daemon --quiet --report-every 25" `
    -WorkingDirectory $root

$evolveTrigger = New-ScheduledTaskTrigger -AtStartup
$evolveSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "Loonie-Evolve" -Action $evolveAction `
    -Trigger $evolveTrigger -Settings $evolveSettings `
    -Description "Continuous strategy search (paper only)" -Force | Out-Null
Write-Host "registered Loonie-Evolve  (at startup, restarts on failure)"

# ---- Trade: one paper rebalance per weekday ------------------------------
$tradeAction = New-ScheduledTaskAction -Execute $PythonExe `
    -Argument "scripts\run_trade.py --flatten-on-halt" `
    -WorkingDirectory $root

$tradeTrigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $TradeTime
$tradeSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "Loonie-Trade" -Action $tradeAction `
    -Trigger $tradeTrigger -Settings $tradeSettings `
    -Description "Daily PAPER rebalance" -Force | Out-Null
Write-Host "registered Loonie-Trade   (weekdays $TradeTime local)"

Write-Host ""
Write-Host "Both tasks run PAPER. Live trading needs three locks this script"
Write-Host "does not touch. Check state with: python scripts\status.py"
Write-Host "Start now:  Start-ScheduledTask -TaskName Loonie-Evolve"
