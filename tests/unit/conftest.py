"""Shared pytest fixtures for the unit test suite.

Currently: isolate the ``OPENARC_FORCE_RECOMPILE`` environment variable. The
``openarc serve start --force-recompile / --fr`` command *sets* this variable
(``os.environ[...] = "true"/"false"``) rather than threading it through an
explicit env map, so the value it sets *leaks* into the process environment and
is not reverted after the command returns. Resetting it to the session baseline
(before and after every test, see the autouse fixture below) keeps the CLI and
server-glue tests that assert on it hermetic regardless of test execution order.

The variable is otherwise only ever inspected by tests of this feature, so the
fixture is a no-op for the rest of the suite.
"""

import os

import pytest

_FORCE_RECOMPILE = "OPENARC_FORCE_RECOMPILE"

# Captured once, when the suite is (still) clean, as the value every test is
# reset to. In practice this is ``None`` (the variable is absent at import), so
# the reset means "ensure it is unset".
_BASELINE = os.environ.get(_FORCE_RECOMPILE)


def _apply_baseline() -> None:
    if _BASELINE is None:
        os.environ.pop(_FORCE_RECOMPILE, None)
    else:
        os.environ[_FORCE_RECOMPILE] = _BASELINE


@pytest.fixture(autouse=True)
def _reset_force_recompile_env():
    # Setup: start every test from the session baseline so a value leaked by a
    # previous test (the command mutates os.environ and is not auto-reverted)
    # cannot influence what this test observes.
    _apply_baseline()
    try:
        yield
    finally:
        # Teardown: put the process back at the session baseline.
        _apply_baseline()
