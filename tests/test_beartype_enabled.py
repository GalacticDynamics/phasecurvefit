"""Test that beartype is enabled during tests."""

import os


def test_beartype_env_var_is_set():
    """Verify that env var is set to beartype.beartype."""
    env_value = os.environ.get("PHASECURVEFIT_ENABLE_RUNTIME_TYPECHECKING")
    assert env_value == "beartype.beartype", (
        f"Expected PHASECURVEFIT_ENABLE_RUNTIME_TYPECHECKING='beartype.beartype', "
        f"got {env_value!r}"
    )
