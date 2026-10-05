# killleft.ps1 - terminate any leftover run-windows-tests.ps1 / its pytest + worker_process
# children (matched by 'run-windows-tests.ps1 / no:cacheprovider / worker_process' in their
# command line) left running after a foreground run whose client-side channel timed out.
# Safe to run: a normal run uses the same tokens, so call this to clean up a BROKEN run
# BEFORE starting a fresh one, not during a healthy one.
$ErrorActionPreference = "Continue"
$pat = 'run-windows-tests\.ps1|no:cacheprovider|worker_process'
$strays = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match $pat }
if (-not $strays) { Write-Output "no stray run-windows-tests / pytest / worker processes"; return }
"$($strays.Count) stray process(es) found; killing:"
foreach ($p in $strays) {
    "  pid=$($p.ProcessId) $p.Name"
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch { "    (kill failed: $($_.Exception.Message))" }
}
Start-Sleep -Milliseconds 500
$after = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match $pat }
if (-not $after) { Write-Output "clean" } else { Write-Output "STILL $($after.Count) after" }
