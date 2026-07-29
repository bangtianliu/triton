// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  ttg.target = "hip:gfx950",
  "ttg.threads-per-warp" = 64 : i32
} {
  // CHECK-LABEL: llvm.func @kernel
  // CHECK-NOT: register-pressure-policy
  tt.func public @kernel() attributes {
    "ttg.amdg.register-pressure-policy" = "minimize-spills"
  } {
    tt.return
  }
}
