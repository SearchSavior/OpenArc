# poll.ps1 - read the output of a detached run-windows-tests.ps1 -Launch.
# Usage:  poll.ps1 -Newest     (or -Run YYYYMMDD-HHmmss)
# Not meant to gate anything; just surface progress / results. 5.1-safe.
[CmdletBinding()]
param(
  [switch]$Newest,
  [string]$Run
)
$ErrorActionPreference = "Continue"
$base = Join-Path $env:TEMP "openarc-logs"
if ($Run) {
  $dir = Join-Path $base ("run-$Run")
} else {
  $hit = Get-ChildItem $base -Filter 'run-*' -Directory -ErrorAction SilentlyContinue |
         Sort-Object Name | Select-Object -Last 1
  $dir = $hit.FullName
}
if (-not $dir -or -not (Test-Path $dir)) {
  Write-Output "NO RUN DIR for run=$Run newest=$Newest under $base"
  exit 1
}
Write-Output "run dir : $dir"
$out  = Join-Path $dir "stdout.log"
$err  = Join-Path $dir "stderr.log"
$done = $false
if (Test-Path $out) {
  $txt = Get-Content $out -Raw -ErrorAction SilentlyContinue
  if ($txt -match 'RUNNER-FINISHED') { $done = $true }
}
Write-Output "DONE    : $done"
Write-Output "================ stdout.log ================"
if (Test-Path $out) {
  Get-Content $out -Raw -ErrorAction SilentlyContinue
} else {
  Write-Output "(stdout.log not present yet)"
}
Write-Output "================ worker .cap log(s) ========"
$caps = Get-ChildItem $dir -Filter '*.cap' -ErrorAction SilentlyContinue
if (-not $caps) {
  Write-Output "(no worker .cap yet -- worker may not have opened its log, or run still going)"
} else {
  foreach ($c in $caps) {
    Write-Output "----- $c.name ($($c.Length) bytes) -----"
    Get-Content $c.FullName -Raw -ErrorAction SilentlyContinue
  }
}
Write-Output "================ stderr.log ================"
if (Test-Path $err) {
  $et = Get-Content $err -Raw -ErrorAction SilentlyContinue
  if (([string]$et).Trim()) { $et } else { Write-Output "(empty)" }
} else {
  Write-Output "(stderr.log not present yet)"
}
Write-Output "============================================"
