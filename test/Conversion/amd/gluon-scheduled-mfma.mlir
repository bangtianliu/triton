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
  // CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"
  // CHECK-NOT: amdg.
  tt.func public @scheduled_mfma() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }

  // A phase-local reduction may span multiple source-scheduled MFMAs. The
  // intermediate result remains on the MFMA dependency chain; only the
  // terminal result crosses the completion boundary.
  // CHECK-LABEL: llvm.func @transient_mfma_chain
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,a,v"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=&v,a,v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"
  // CHECK-NOT: amdg.
  tt.func public @transient_mfma_chain() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %partial = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %partial
        resident "rhs" accumulator "transient" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }

  // Two independent result fragments share one completion boundary. All
  // result fragments and the live dependency are tied through its outputs.
  // CHECK-LABEL: llvm.func @multi_fragment_commit
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,a,v"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,a,v"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 5", "=v,=v,=a,0,1,2,~{memory}"
  // CHECK-NOT: amdg.
  tt.func public @multi_fragment_commit() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result0 = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %result1 = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %committed0, %committed1, %preserved =
        amdg.mfma_commit %result0, %result1, %b
        : tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma>,
          tensor<32x16xbf16, #rhs>
    tt.return
  }

  // CHECK-LABEL: llvm.func @persistent_accumulator
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,v,0"
  // CHECK-NOT: amdg.
  tt.func public @persistent_accumulator() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // A resident operand and a persistent accumulator may both use matrix
  // storage. Distinct input constraints allow the allocator to assign them
  // independently without imposing an invalid early-clobber restriction.
  // CHECK-LABEL: llvm.func @resident_matrix
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,a,0"
  // CHECK-NOT: amdg.
  tt.func public @resident_matrix() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "lhs" accumulator "persistent" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }
}
