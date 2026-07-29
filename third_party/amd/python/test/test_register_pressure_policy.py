from pathlib import Path

import pytest
import triton

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)

POLICY_TTIR = str(Path(__file__).parent / "register_pressure_policy.ttir")


def test_register_pressure_policy_is_compiler_metadata():
    target = triton.runtime.driver.active.get_current_target()
    compiled = triton.compile(POLICY_TTIR, target=target)

    assert compiled.metadata.register_pressure_policy == "minimize-spills"
    assert "register-pressure-policy" not in compiled.asm["llir"]
