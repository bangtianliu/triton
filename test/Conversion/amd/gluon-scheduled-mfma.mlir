// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  ttg.target = "hip:gfx950",
  "ttg.threads-per-warp" = 64 : i32
} {
  // CHECK-LABEL: llvm.func @scheduled_mfma
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,a,v"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 5", "=v,0,a,~{memory}"
  // CHECK-NOT: amdg.
  tt.func public @scheduled_mfma() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "vector" initialize true commit false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %committed = amdg.mfma_commit %result preserve %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
          -> tensor<16x16xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func @persistent_accumulator
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 5", "=a,0,~{memory}"
  // CHECK-NOT: amdg.
  tt.func public @persistent_accumulator() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "matrix" initialize false commit true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }
}
