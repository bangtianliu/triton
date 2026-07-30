from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import triton

from triton._C.libtriton import llvm
from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)

POLICY_TTIR = str(Path(__file__).parent / "register_pressure_policy.ttir")


def test_register_pressure_policy_is_compiler_metadata():
    target = triton.runtime.driver.active.get_current_target()
    compiled = triton.compile(POLICY_TTIR, target=target)

    assert compiled.metadata.register_pressure_policy == "minimize-spills"
    assert "register-pressure-policy" not in compiled.asm["llir"]


def test_codegen_flags_are_scoped_per_compilation(capfd):
    target = triton.runtime.driver.active.get_current_target()
    llvm.init_targets()
    llvm_ir = """
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @kernel() {
entry:
  ret void
}
    """

    def translate(flags):
        llvm.translate_to_asm(
            llvm_ir,
            "amdgcn-amd-amdhsa",
            target.arch,
            "",
            flags,
            False,
            False,
            False,
        )
        return capfd.readouterr().err

    policy_flags = ["sink-insts-to-avoid-spills", "print-after-all"]
    first_flagged = translate(policy_flags)
    ordinary = translate([])
    second_flagged = translate(policy_flags)

    assert "IR Dump" in first_flagged
    assert "IR Dump" not in ordinary
    assert "IR Dump" in second_flagged
    assert "may only occur zero or one times" not in (
        first_flagged + ordinary + second_flagged
    )


def test_codegen_flags_are_thread_safe(capfd):
    target = triton.runtime.driver.active.get_current_target()
    llvm.init_targets()
    llvm_ir = """
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @kernel() {
entry:
  ret void
}
    """
    num_workers = 8
    barrier = Barrier(num_workers)

    def translate(index):
        barrier.wait()
        flags = ["sink-insts-to-avoid-spills"] if index % 2 else []
        return llvm.translate_to_asm(
            llvm_ir,
            "amdgcn-amd-amdhsa",
            target.arch,
            "",
            flags,
            False,
            False,
            False,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        outputs = list(pool.map(translate, range(num_workers)))

    assert all("kernel" in output for output in outputs)
    assert "may only occur zero or one times" not in capfd.readouterr().err
