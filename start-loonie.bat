@echo off
REM ===================================================================
REM  loonie - one button to bring everything back up.
REM
REM  Double-click this, or use the desktop shortcut created by
REM  scripts\install_shortcut.ps1. Safe to run twice: it checks for an
REM  already-running search before starting another one.
REM
REM  Starts:
REM    1. the strategy search daemon  (resumes from saved state)
REM    2. the dashboard web server    (localhost + your LAN)
REM  and opens the dashboard in your browser.
REM
REM  PAPER TRADING ONLY. Nothing here can arm real money.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo.
echo  loonie - starting
echo  ---------------------------------------------------------------

REM --- 1. search daemon (skip if one is already running) -------------
tasklist /FI "WINDOWTITLE eq loonie-search" 2>NUL | find /I "cmd.exe" >NUL
if errorlevel 1 (
    echo  [1/2] starting strategy search ^(resumes from saved state^)
    start "loonie-search" /MIN cmd /c "python scripts\run_evolve.py --daemon --quiet --report-every 50 >> state\evolve.log 2>&1"
) else (
    echo  [1/2] strategy search already running
)

REM --- 2. dashboard ---------------------------------------------------
echo  [2/2] starting dashboard
echo.

REM --tunnel adds a public https URL (needs cloudflared installed).
REM Add it here if you want the dashboard reachable off your network.
python scripts\serve.py %*

endlocal
