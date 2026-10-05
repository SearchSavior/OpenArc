# readlogs.ps1 - dump the per-model worker log(s) from the LAST run, so a load
# timeout's cause is visible. The worker writes its own log DIRECTLY (bypassing the
# launching shell's buffering), so it persists on disk after the run even though the
# supervisor drains the worker pipe and forwards nothing to the main log.
# Robust/idiomatic pipeline form (Get-ChildItem | Sort-Object | ForEach-Object) so it
# works identically under Windows PowerShell 5.1 and 7. Prints a file in full if it is
# small, otherwise head+tail.
param([string]$Dir = "$env:TEMP\openarc-logs", [int]$LargeMin = 16000)
$ErrorActionPreference = "Continue"
"logs dir : $Dir"

$logs = Get-ChildItem $Dir -Filter '*.log' -ErrorAction SilentlyContinue | Sort-Object Name
if (-not $logs) {
  Write-Output "  (no *.log in dir -- run may not have written a worker log / may have been cleared)"
}
foreach ($f in $logs) {
  Write-Output ""
  Write-Output "=================================================================="
  Write-Output "----- $($f.Name)  ($($f.Length) bytes)  $($f.LastWriteTime.ToString('HH:mm:ss')) -----"
  Write-Output "=================================================================="
  if ($f.Length -lt $LargeMin) {
    Get-Content $f.FullName -Raw -ErrorAction SilentlyContinue
  } else {
    Write-Output "----- HEAD -------------------"
    Get-Content $f.FullName -TotalCount 25 -ErrorAction SilentlyContinue | ForEach-Object { $_ }
    Write-Output "----- TAIL (last 25 lines) -----"
    Get-Content $f.FullName -Tail 25 -ErrorAction SilentlyContinue | ForEach-Object { $_ }
  }
}

Write-Output ""
Write-Output "===== worker process(es) still alive ====="
$alive = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'worker_process|OPENARC_WORKER' }
if (-not $alive) {
  Write-Output "  (none) -- no worker child still running"
} else {
  foreach ($p in $alive) {
    $cl = $p.CommandLine
    if ($cl.Length -gt 130) { $cl = $cl.Substring(0,130) }
    Write-Output "  LIVE pid=$($p.ProcessId) $p.Name :: $cl"
  }
}
