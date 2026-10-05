# run-windows-tests.ps1
# Run from the repo toplevel:  pwsh -File util/pwsh/run-windows-tests.ps1   (add -CollectOnly to only collect)
# Runs ONLY the 3 worker-subprocess tests from the last commit, streaming pytest to ssh output.
# -p no:cacheprovider keeps pytest from writing a .pytest_cache into the read-only W: mirror.
# -o addopts= / -o testpaths= clear the project's own config, because once pytest-asyncio is
# installed it makes pytest sweep the whole tests/ tree even when you name 3 files (34 -> 791,
# then a 300s time-out); clearing the two project keys lets the explicit files scope collection.
[CmdletBinding()]
param(
  [switch]$CollectOnly
)
$tests = @(
  'tests/unit/test_worker_platform_unit.py',
  'tests/unit/test_worker_log_file_unit.py',
  'tests/integration/test_worker_windows_integration.py::test_stub_vlm_worker_full_lifecycle_on_windows'
)
#$margs = @('-m', 'pytest', '-o', 'addopts=', '-o', 'testpaths=', '-p', 'no:cacheprovider', '-v') + $tests
$margs = @('-m', 'pytest', '-p', 'no:cacheprovider', '-v') + $tests
if ($CollectOnly) { $margs += '--co' }
& python @margs
"pytest exit = $LASTEXITCODE"
