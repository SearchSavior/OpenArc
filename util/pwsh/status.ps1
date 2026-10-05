# status.ps1 - live status of the newest detached run-windows-tests.ps1 run.
# Shows process liveness, the (worker-written) per-run log files, stdout.log tail,
# and DONE state. 5.1-safe; invoked via -File so nothing is shell-quoting-fragile.
[CmdletBinding()]
param([int]$MaxTail=50)
$ErrorActionPreference = "Continue"
$base = Join-Path $env:TEMP "openarc-logs"
$hit = Get-ChildItem $base -Filter 'run-*' -Directory -ErrorAction SilentlyContinue |
       Sort-Object Name | Select-Object -Last 1
if (-not $hit) { Write-Output "no run dirs under $base"; exit 1 }
$dir = $hit.FullName
Write-Output "NEWEST RUN DIR: $dir"
Write-Output ""
Write-Output "= processes = (looking for our runner + worker children)"
$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'run-windows-tests|worker_process|pytest|openarc' }
if (-not $procs) {
  Write-Output "  (none) -- no runner/worker process currently alive"
} else {
  foreach ($p in $procs) {
    $cl = $p.CommandLine
    if ($cl.Length -gt 160) { $cl = $cl.Substring(0,160) }
    Write-Output ("  pid={0,-6} {1,-14} :: {2}" -f $p.ProcessId, $p.Name, $cl)
  }
}
Write-Output ""
Write-Output "= all files under base (recursive) = "
Get-ChildItem $base -Recurse -Force -ErrorAction SilentlyContinue |
  ForEach-Object { ("  {0,-60} {1,8}  {2}" -f $_.FullName, $_.Length, $_.LastWriteTime) }
Write-Output ""
# The worker writes its OWN log here directly (not via the launcher's redirection),
# so these show progress even while the launcher's stdout is still buffered.
Write-Output "= live worker-written *.log in this run dir = "
$wlogs = Get-ChildItem $dir -Filter '*.log' -NotRecurse -ErrorAction SilentlyContinue
if (-not $wlogs) { Write-Output "  (none -- worker has not opened a log yet)" } else {
  foreach ($w in $wlogs) {
    Write-Output "----- $($w.Name) ($($w.Length) bytes) -----"
    $c = Get-Content $w.FullName -Tail $MaxTail -ErrorAction SilentlyContinue
    if ($c) { $c | ForEach-Object { $_ } } else { "  (empty file)" }
  }
}
Write-Output ""
Write-Output "= stdout.log tail (buffered: only appears once the run finishes) = "
$out = Join-Path $dir "stdout.log"
$done = $false
if (Test-Path $out) {
  $raw = Get-Content $out -Raw -ErrorAction SilentlyContinue
  if ($raw -match 'RUNNER-FINISHED') { $done = $true }
  $lines = @($raw -split "`r?`n")
  $n = $lines.Count
  $s = if ($n -gt $MaxTail) { $n - $MaxTail } else { 0 }
  Write-Output ("  (stdout.log has {0} lines; DONE={1})" -f $n, $done)
  for ($i=$s; $i -lt $n; $i++) { $lines[$i] }
} else {
  Write-Output "  (stdout.log not created; launcher may have failed to start)"
}
