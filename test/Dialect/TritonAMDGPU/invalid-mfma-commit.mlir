// RUN: triton-opt %s --verify-diagnostics

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#mma_other = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  ttg.target = "hip:gfx950",
  "ttg.threads-per-warp" = 64 : i32
} {
  tt.func public @reject_matrix_source() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %matrix = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    // expected-error@+1 {{input 0 must be a direct transient scheduled_mfma result}}
    %result, %preserved = amdg.mfma_commit %matrix, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }

  tt.func public @reject_cross_lane_conversion() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error@+1 {{transient result must be consumed by amdg.mfma_commit or as the accumulator of another transient amdg.scheduled_mfma}}
    %vector = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %converted = ttg.convert_layout %vector
        : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma_other>
    %result, %preserved = amdg.mfma_commit %converted, %b
        : tensor<16x16xf32, #mma_other>, tensor<32x16xbf16, #rhs>
    tt.return
  }

  tt.func public @reject_use_before_boundary() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error@+1 {{transient result must have exactly one completion-chain use}}
    %vector = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %used = arith.addf %vector, %vector
        : tensor<16x16xf32, #mma>
    %result, %preserved = amdg.mfma_commit %vector, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }

  tt.func public @reject_transient_without_boundary() {
    %a = arith.constant dense<1.000000e+00> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<2.000000e+00> : tensor<32x16xbf16, #rhs>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error@+1 {{transient result must be consumed by amdg.mfma_commit or as the accumulator of another transient amdg.scheduled_mfma}}
    %vector = amdg.scheduled_mfma %a, %b, %acc
        resident "rhs" accumulator "transient" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %used = arith.addf %vector, %acc : tensor<16x16xf32, #mma>
    tt.return
  }
}
