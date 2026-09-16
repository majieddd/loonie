<#
    Put a "loonie" button on the Desktop, and optionally start everything
    automatically when you log in.

        powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
        powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -AtLogon
        powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -Remove

    -AtLogon registers a Task Scheduler entry so that after a reboot or power
    cut the search and dashboard come back on their own. That is the setting
    that matters for a machine meant to stay up: the button is for when you
    want it back *now*, the logon task is for when nobody is watching.

    PAPER only. Neither the shortcut nor the task passes the live-trading flag.
#>

param(
    [switch]$AtLogon,
    [switch]$Remove,
    [switch]$Tunnel
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bat  = Join-Path $root "start-loonie.bat"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "loonie.lnk"
$taskName = "Loonie-AtLogon"

if ($Remove) {
    if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Host "removed desktop shortcut" }
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "removed logon task"
    }
    exit 0
}

if (-not (Test-Path $bat)) { throw "start-loonie.bat not found at $bat" }

# ---- desktop shortcut ----------------------------------------------------
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath       = $bat
if ($Tunnel) { $sc.Arguments = "--tunnel" }
$sc.WorkingDirectory = $root
$sc.Description      = "Start the loonie strategy search and dashboard (paper only)"
$sc.IconLocation     = "$env:SystemRoot\System32\SHELL32.dll,23"
$sc.Save()
Write-Host "desktop shortcut : $lnk"

# ---- optional: start at logon -------------------------------------------
if ($AtLogon) {
    $args = if ($Tunnel) { "--tunnel" } else { "" }
    $action = New-ScheduledTaskAction -Execute $bat -Argument $args -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 2) `
        -ExecutionTimeLimit (New-TimeSpan -Days 0) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Description "Start loonie search + dashboard at logon (paper only)" `
        -Force | Out-Null
    Write-Host "logon task       : $taskName (survives reboots)"
}

Write-Host ""
Write-Host "Double-click 'loonie' on your Desktop to start the search and dashboard."
if (-not $AtLogon) {
    Write-Host "To also start it automatically after a reboot, re-run with -AtLogon"
}
