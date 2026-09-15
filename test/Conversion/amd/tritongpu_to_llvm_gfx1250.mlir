// RUN:  triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch="gfx1250" | FileCheck %s --check-prefix=GFX1250

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @tdm_store_zero_offset_subslice
  tt.func @tdm_store_zero_offset_subslice(%desc: !tt.tensordesc<16x64xf16, #shared>, %input: tensor<32x64xf16, #blocked>) {
    %alloc = ttg.local_alloc %input : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[0, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"
    amdg.async_tdm_copy_local_to_global %desc from %view : !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64> -> !tt.tensordesc<16x64xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %wait = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }

  // GFX1250-LABEL: @tdm_load_store_runtime_subslice
  tt.func @tdm_load_store_runtime_subslice(%desc: !tt.tensordesc<16x64xf16, #shared>, %row_raw: i32) {
    %c16 = arith.constant 16 : i32
    %row = arith.andi %row_raw, %c16 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[%row, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: %[[ROW_MASK:.*]] = llvm.mlir.constant(31 : i32)
    // GFX1250: %[[ROWS:.*]] = llvm.and %{{.*}}, %[[ROW_MASK]] : i32
    // GFX1250: %[[ROW_SHIFT:.*]] = llvm.mlir.constant(6 : i32)
    // GFX1250: %[[ROW_OFFSET:.*]] = llvm.shl %[[ROWS]], %[[ROW_SHIFT]] : i32
    // GFX1250: %[[LAYOUT_OFFSET:.*]] = llvm.or disjoint %[[ROW_OFFSET]],
    // GFX1250: %[[AFFINE_OFFSET:.*]] = llvm.or disjoint %[[LAYOUT_OFFSET]],
    // GFX1250: %[[LOAD_OFFSET:.*]] = llvm.xor %{{.*}}, %[[AFFINE_OFFSET]] : i32
    // GFX1250: %[[LOAD_BASE:.*]] = llvm.getelementptr %{{.*}}[%[[LOAD_OFFSET]]] {{.*}}, f16
    // GFX1250: %[[LOAD_PTR:.*]] = llvm.getelementptr %[[LOAD_BASE]]
    // GFX1250: %[[LOAD_ADDR:.*]] = llvm.ptrtoint %[[LOAD_PTR]] : !llvm.ptr<3> to i32
    // GFX1250: %[[LOAD_DESC:.*]] = llvm.insertelement %[[LOAD_ADDR]],
    // GFX1250: %[[LOAD_I64:.*]] = llvm.bitcast %[[LOAD_DESC]] : vector<4xi32> to vector<2xi64>
    // GFX1250: %[[LOAD_UPDATED:.*]] = llvm.insertelement %{{.*}}, %[[LOAD_I64]]
    // GFX1250: %[[LOAD_FINAL:.*]] = llvm.bitcast %[[LOAD_UPDATED]] : vector<2xi64> to vector<4xi32>
    // GFX1250: "llvm.amdgcn.tensor.load.to.lds"(%[[LOAD_FINAL]],
    %load = amdg.async_tdm_copy_global_to_local %desc into %view : !tt.tensordesc<16x64xf16, #shared> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %loaded = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    // GFX1250: %[[STORE_BASE:.*]] = llvm.getelementptr {{.*}}, f16
    // GFX1250: %[[STORE_PTR:.*]] = llvm.getelementptr %[[STORE_BASE]]
    // GFX1250: %[[STORE_ADDR:.*]] = llvm.ptrtoint %[[STORE_PTR]] : !llvm.ptr<3> to i32
    // GFX1250: %[[STORE_DESC:.*]] = llvm.insertelement %[[STORE_ADDR]],
    // GFX1250: %[[STORE_I64:.*]] = llvm.bitcast %[[STORE_DESC]] : vector<4xi32> to vector<2xi64>
    // GFX1250: %[[STORE_UPDATED:.*]] = llvm.insertelement %{{.*}}, %[[STORE_I64]]
    // GFX1250: %[[STORE_FINAL:.*]] = llvm.bitcast %[[STORE_UPDATED]] : vector<2xi64> to vector<4xi32>
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"(%[[STORE_FINAL]],
    amdg.async_tdm_copy_local_to_global %desc from %view : !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64> -> !tt.tensordesc<16x64xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %stored = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }

  // GFX1250-LABEL: @tdm_load_store_one_row_zero_offset_subslice
  tt.func @tdm_load_store_one_row_zero_offset_subslice(%desc: !tt.tensordesc<1x16xf16, #shared>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[0, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: "llvm.amdgcn.tensor.load.to.lds"
    %load = amdg.async_tdm_copy_global_to_local %desc into %view : !tt.tensordesc<1x16xf16, #shared> -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %loaded = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"
    amdg.async_tdm_copy_local_to_global %desc from %view : !ttg.memdesc<1x16xf16, #shared, #smem, mutable, 32x64> -> !tt.tensordesc<1x16xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %stored = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }

  // GFX1250-LABEL: @tdm_load_store_one_row_runtime_subslice
  tt.func @tdm_load_store_one_row_runtime_subslice(%desc: !tt.tensordesc<1x16xf16, #shared>, %row_raw: i32) {
    %c31 = arith.constant 31 : i32
    %row = arith.andi %row_raw, %c31 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[%row, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: llvm.and
    // GFX1250: %[[ONE_ROW_MASK:.*]] = llvm.mlir.constant(31 : i32)
    // GFX1250: %[[ONE_ROW:.*]] = llvm.and %{{.*}}, %[[ONE_ROW_MASK]] : i32
    // GFX1250: %[[ONE_ROW_SHIFT:.*]] = llvm.mlir.constant(6 : i32)
    // GFX1250: llvm.shl %[[ONE_ROW]], %[[ONE_ROW_SHIFT]] : i32
    // GFX1250: "llvm.amdgcn.tensor.load.to.lds"
    %load = amdg.async_tdm_copy_global_to_local %desc into %view : !tt.tensordesc<1x16xf16, #shared> -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %loaded = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"
    amdg.async_tdm_copy_local_to_global %desc from %view : !ttg.memdesc<1x16xf16, #shared, #smem, mutable, 32x64> -> !tt.tensordesc<1x16xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %stored = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @tdm_store_leading_unit_zero_offset_subslice
  tt.func private @tdm_store_leading_unit_zero_offset_subslice(%desc: !tt.tensordesc<1x16x64xf16, #shared>, %src: !ttg.memdesc<1x32x64xf16, #shared, #smem>) {
    %view = ttg.memdesc_subslice %src[0, 0, 0] : !ttg.memdesc<1x32x64xf16, #shared, #smem> -> !ttg.memdesc<1x16x64xf16, #shared, #smem, 1x32x64>
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"
    amdg.async_tdm_copy_local_to_global %desc from %view : !ttg.memdesc<1x16x64xf16, #shared, #smem, 1x32x64> -> !tt.tensordesc<1x16x64xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %wait = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }

  // GFX1250-LABEL: @tdm_store_leading_unit_runtime_subslice
  tt.func private @tdm_store_leading_unit_runtime_subslice(%desc: !tt.tensordesc<1x16x64xf16, #shared>, %src: !ttg.memdesc<1x32x64xf16, #shared, #smem>, %row_raw: i32) {
    %c16 = arith.constant 16 : i32
    %row = arith.andi %row_raw, %c16 : i32
    %view = ttg.memdesc_subslice %src[0, %row, 0] : !ttg.memdesc<1x32x64xf16, #shared, #smem> -> !ttg.memdesc<1x16x64xf16, #shared, #smem, 1x32x64>
    // GFX1250: %[[UNIT_ROW_MASK:.*]] = llvm.mlir.constant(31 : i32)
    // GFX1250: %[[UNIT_ROWS:.*]] = llvm.and %{{.*}}, %[[UNIT_ROW_MASK]] : i32
    // GFX1250: %[[UNIT_ROW_SHIFT:.*]] = llvm.mlir.constant(6 : i32)
    // GFX1250: %[[UNIT_ROW_OFFSET:.*]] = llvm.shl %[[UNIT_ROWS]], %[[UNIT_ROW_SHIFT]] : i32
    // GFX1250: %[[UNIT_LAYOUT_OFFSET:.*]] = llvm.or disjoint %[[UNIT_ROW_OFFSET]],
    // GFX1250: %[[UNIT_AFFINE_OFFSET:.*]] = llvm.or disjoint %[[UNIT_LAYOUT_OFFSET]],
    // GFX1250: %[[UNIT_OFFSET:.*]] = llvm.xor %{{.*}}, %[[UNIT_AFFINE_OFFSET]] : i32
    // GFX1250: %[[UNIT_BASE:.*]] = llvm.getelementptr %{{.*}}[%[[UNIT_OFFSET]]] {{.*}}, f16
    // GFX1250: %[[UNIT_PTR:.*]] = llvm.getelementptr %[[UNIT_BASE]]
    // GFX1250: %[[UNIT_ADDR:.*]] = llvm.ptrtoint %[[UNIT_PTR]] : !llvm.ptr<3> to i32
    // GFX1250: %[[UNIT_DESC:.*]] = llvm.insertelement %[[UNIT_ADDR]],
    // GFX1250: %[[UNIT_I64:.*]] = llvm.bitcast %[[UNIT_DESC]] : vector<4xi32> to vector<2xi64>
    // GFX1250: %[[UNIT_UPDATED:.*]] = llvm.insertelement %{{.*}}, %[[UNIT_I64]]
    // GFX1250: %[[UNIT_FINAL:.*]] = llvm.bitcast %[[UNIT_UPDATED]] : vector<2xi64> to vector<4xi32>
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"(%[[UNIT_FINAL]],
    amdg.async_tdm_copy_local_to_global %desc from %view : !ttg.memdesc<1x16x64xf16, #shared, #smem, 1x32x64> -> !tt.tensordesc<1x16x64xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %wait = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }
}

// -----

#indices = #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @tdm_gather_scatter_static_subslice
  tt.func @tdm_gather_scatter_static_subslice(%desc: !tt.tensordesc<8x64xf16, #shared>, %indices: tensor<8xi32, #indices>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %alloc[16, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<8x64xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: %[[GATHER_BASE:.*]] = llvm.getelementptr {{.*}}, f16
    // GFX1250: %[[GATHER_PTR:.*]] = llvm.getelementptr %[[GATHER_BASE]]
    // GFX1250: %[[GATHER_ADDR:.*]] = llvm.ptrtoint %[[GATHER_PTR]] : !llvm.ptr<3> to i32
    // GFX1250: %[[GATHER_DESC:.*]] = llvm.insertelement %[[GATHER_ADDR]],
    // GFX1250: "llvm.amdgcn.tensor.load.to.lds"(%[[GATHER_DESC]],
    %gather = amdg.async_tdm_gather %desc[%indices] to %view : tensor<8xi32, #indices>, !ttg.memdesc<8x64xf16, #shared, #smem, mutable, 32x64> -> !tt.tensordesc<8x64xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %gathered = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    // GFX1250: %[[SCATTER_BASE:.*]] = llvm.getelementptr {{.*}}, f16
    // GFX1250: %[[SCATTER_PTR:.*]] = llvm.getelementptr %[[SCATTER_BASE]]
    // GFX1250: %[[SCATTER_ADDR:.*]] = llvm.ptrtoint %[[SCATTER_PTR]] : !llvm.ptr<3> to i32
    // GFX1250: %[[SCATTER_DESC:.*]] = llvm.insertelement %[[SCATTER_ADDR]],
    // GFX1250: "llvm.amdgcn.tensor.store.from.lds"(%[[SCATTER_DESC]],
    amdg.async_tdm_scatter %desc[%indices] from %view : tensor<8xi32, #indices>, !ttg.memdesc<8x64xf16, #shared, #smem, mutable, 32x64> -> !tt.tensordesc<8x64xf16, #shared>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %scattered = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @tdm_fused_static_runtime_subslices
  tt.func @tdm_fused_static_runtime_subslices(%desc0: !tt.tensordesc<16x64xf16, #shared>, %desc1: !tt.tensordesc<16x64xf16, #shared>, %row_raw: i32) {
    %c16 = arith.constant 16 : i32
    %row = arith.andi %row_raw, %c16 : i32
    %alloc0 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %alloc1 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %view0 = ttg.memdesc_subslice %alloc0[16, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>
    %view1 = ttg.memdesc_subslice %alloc1[%row, 0] : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: %[[FUSED_BASE0:.*]] = llvm.getelementptr {{.*}}, f16
    // GFX1250: %[[FUSED_BASE1:.*]] = llvm.getelementptr {{.*}}, f16
    // GFX1250: %[[FUSED_PTR0:.*]] = llvm.getelementptr %[[FUSED_BASE0]]
    // GFX1250: %[[FUSED_ADDR0:.*]] = llvm.ptrtoint %[[FUSED_PTR0]] : !llvm.ptr<3> to i32
    // GFX1250: %[[FUSED_DESC0:.*]] = llvm.insertelement %[[FUSED_ADDR0]],
    // GFX1250: %[[FUSED_I64_0:.*]] = llvm.bitcast %[[FUSED_DESC0]] : vector<4xi32> to vector<2xi64>
    // GFX1250: %[[FUSED_UPDATED0:.*]] = llvm.insertelement %{{.*}}, %[[FUSED_I64_0]]
    // GFX1250: %[[FUSED_FINAL0:.*]] = llvm.bitcast %[[FUSED_UPDATED0]] : vector<2xi64> to vector<4xi32>
    // GFX1250: %[[FUSED_PTR1:.*]] = llvm.getelementptr %[[FUSED_BASE1]]
    // GFX1250: %[[FUSED_ADDR1:.*]] = llvm.ptrtoint %[[FUSED_PTR1]] : !llvm.ptr<3> to i32
    // GFX1250: %[[FUSED_DESC1:.*]] = llvm.insertelement %[[FUSED_ADDR1]],
    // GFX1250: %[[FUSED_I64_1:.*]] = llvm.bitcast %[[FUSED_DESC1]] : vector<4xi32> to vector<2xi64>
    // GFX1250: %[[FUSED_UPDATED1:.*]] = llvm.insertelement %{{.*}}, %[[FUSED_I64_1]]
    // GFX1250: %[[FUSED_FINAL1:.*]] = llvm.bitcast %[[FUSED_UPDATED1]] : vector<2xi64> to vector<4xi32>
    // GFX1250: %[[FUSED_SELECTED:.*]] = llvm.select %{{.*}}, %[[FUSED_FINAL0]], %[[FUSED_FINAL1]] : i1, vector<4xi32>
    // GFX1250: "llvm.amdgcn.tensor.load.to.lds"(%[[FUSED_SELECTED]],
    %fused = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %view0, %view1 {warp_used_hints = array<i32: 1, 2>} : !tt.tensordesc<16x64xf16, #shared>, !tt.tensordesc<16x64xf16, #shared> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>, !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 32x64>
    // GFX1250: rocdl.s.wait.tensorcnt 0
    %wait = amdg.async_tdm_intrinsic_wait {count = 0 : i32}
    tt.return
  }
}

// -----

#dynamic_inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dynamic_partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #dynamic_inner}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @dynamic_subslice_partitioned_prefix
  tt.func private @dynamic_subslice_partitioned_prefix(%src: !ttg.memdesc<5x16x16xf16, #dynamic_partitioned, #ttg.shared_memory>, %stage: i32, %row: i32) -> !ttg.memdesc<2x8x16xf16, #dynamic_partitioned, #ttg.shared_memory, 5x16x16> {
    // GFX1250: %[[STRIDE:.*]] = llvm.mlir.constant(128 : i32)
    // GFX1250: %[[PREFIX:.*]] = llvm.mul %arg1, %[[STRIDE]] : i32
    // GFX1250-COUNT-2: llvm.getelementptr %{{.*}}[%[[PREFIX]]]
    // GFX1250: llvm.insertelement
    // GFX1250: %[[BASES:.*]] = llvm.insertelement
    // GFX1250: %[[SELECT0:.*]] = llvm.xor %{{.*}}, %{{.*}} : i32
    // GFX1250-NEXT: %[[BASE0:.*]] = llvm.extractelement %[[BASES]][%[[SELECT0]] : i32] : vector<2x!llvm.ptr<3>>
    // GFX1250: %[[SELECT1:.*]] = llvm.xor %{{.*}}, %{{.*}} : i32
    // GFX1250-NEXT: %[[BASE1:.*]] = llvm.extractelement %[[BASES]][%[[SELECT1]] : i32] : vector<2x!llvm.ptr<3>>
    // GFX1250: llvm.insertvalue %[[BASE0]],
    // GFX1250: llvm.insertvalue %[[BASE1]],
    %view = ttg.memdesc_subslice %src[%stage, %row, 0] : !ttg.memdesc<5x16x16xf16, #dynamic_partitioned, #ttg.shared_memory> -> !ttg.memdesc<2x8x16xf16, #dynamic_partitioned, #ttg.shared_memory, 5x16x16>
    tt.return %view : !ttg.memdesc<2x8x16xf16, #dynamic_partitioned, #ttg.shared_memory, 5x16x16>
  }

  // GFX1250-LABEL: @subslice_partitioned_runtime_prefix_static_row
  tt.func private @subslice_partitioned_runtime_prefix_static_row(%src: !ttg.memdesc<5x16x16xf16, #dynamic_partitioned, #ttg.shared_memory>, %stage: i32) -> !ttg.memdesc<2x8x16xf16, #dynamic_partitioned, #ttg.shared_memory, 5x16x16> {
    // GFX1250: %[[STATIC_BASE0:.*]] = llvm.extractvalue %arg0[0]
    // GFX1250: %[[STATIC_BASE1:.*]] = llvm.extractvalue %arg0[1]
    // GFX1250: %[[STATIC_STRIDE:.*]] = llvm.mlir.constant(128 : i32)
    // GFX1250: %[[STATIC_PREFIX:.*]] = llvm.mul %arg1, %[[STATIC_STRIDE]] : i32
    // GFX1250: %[[ADVANCED0:.*]] = llvm.getelementptr %[[STATIC_BASE0]][%[[STATIC_PREFIX]]]
    // GFX1250: %[[ADVANCED1:.*]] = llvm.getelementptr %[[STATIC_BASE1]][%[[STATIC_PREFIX]]]
    // GFX1250-NOT: llvm.extractelement
    // GFX1250: %[[ROTATED0:.*]] = llvm.insertvalue %[[ADVANCED1]], %{{.*}}[0]
    // GFX1250-NEXT: %[[ROTATED1:.*]] = llvm.insertvalue %[[ADVANCED0]], %[[ROTATED0]][1]
    // GFX1250-NOT: llvm.extractelement
    // GFX1250: llvm.return
    %view = ttg.memdesc_subslice %src[%stage, 8, 0] : !ttg.memdesc<5x16x16xf16, #dynamic_partitioned, #ttg.shared_memory> -> !ttg.memdesc<2x8x16xf16, #dynamic_partitioned, #ttg.shared_memory, 5x16x16>
    tt.return %view : !ttg.memdesc<2x8x16xf16, #dynamic_partitioned, #ttg.shared_memory, 5x16x16>
  }
}

// -----

#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @experimental_dynamic_partitioned_memdesc_to_i32
  tt.func private @experimental_dynamic_partitioned_memdesc_to_i32(%row: i32) -> i32 {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    // Rows 0, 4, 8, and 12 select both partitions and intra-partition byte
    // offsets 0 and 128. The first rotated base is the logical origin.
    // GFX1250: llvm.extractelement
    %view = ttg.memdesc_subslice %alloc[%row, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    // GFX1250: %[[ORIGIN:.*]] = llvm.extractvalue %{{.*}}[0]
    // GFX1250: %[[INNER_MASK:.*]] = llvm.mlir.constant(7 : i32)
    // GFX1250: %[[INNER_ROW:.*]] = llvm.and %{{.*}}, %[[INNER_MASK]] : i32
    // GFX1250: %[[ROW_SHIFT:.*]] = llvm.mlir.constant(4 : i32)
    // GFX1250: llvm.shl %[[INNER_ROW]], %[[ROW_SHIFT]] : i32
    // GFX1250: %[[ELEM_BYTES:.*]] = llvm.mlir.constant(2 : i32)
    // GFX1250: %[[BYTES:.*]] = llvm.mul %{{.*}}, %[[ELEM_BYTES]] : i32
    // GFX1250: %[[BASE_ADDRESS:.*]] = llvm.ptrtoint %[[ORIGIN]] : !llvm.ptr<3> to i32
    // GFX1250: %[[ADDRESS:.*]] = llvm.add %[[BYTES]], %[[BASE_ADDRESS]] : i32
    // GFX1250: %[[KEY:.*]] = llvm.and %[[ADDRESS]], %{{.*}} : i32
    %address = tti.experimental_memdesc_to_i32 %view : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    // GFX1250: llvm.return %[[KEY]]
    tt.return %address : i32
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4]], warp = [[16, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: wmma_permlane16_swap
  tt.func @wmma_permlane16_swap(%arg0: tensor<32x32xf16, #mma>) {
    // GFX1250-NOT: store
    // GFX1250-NOT: load
    // GFX1250-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    // GFX1250-NOT: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#noncontiguous = #ttg.generic_linear<{register = [[0, 1]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: generic_linear_noncontiguous_fp32_to_bf16
  tt.func @generic_linear_noncontiguous_fp32_to_bf16(%arg0: tensor<8x8xf32, #noncontiguous>) -> tensor<8x8xbf16, #noncontiguous> {
    // GFX1250-NOT: llvm.call_intrinsic "llvm.amdgcn.perm"
    // GFX1250-COUNT-2: llvm.trunc {{.*}} : i32 to i16
    // GFX1250-NOT: llvm.trunc
    // GFX1250-NOT: llvm.call_intrinsic "llvm.amdgcn.perm"
    %0 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #noncontiguous> -> tensor<8x8xbf16, #noncontiguous>
    tt.return %0 : tensor<8x8xbf16, #noncontiguous>
  }
}

// -----

#partition_aware = #ttg.generic_linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32], [16, 0], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4]], warp = [[64, 64], [32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: generic_linear_fp32_to_fp8
  tt.func @generic_linear_fp32_to_fp8(%arg0: tensor<128x256xf32, #partition_aware>) -> tensor<128x256xf8E4M3FN, #partition_aware> {
    // GFX1250-COUNT-16: rocdl.cvt.scalef32.pk8.fp8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128x256xf32, #partition_aware> -> tensor<128x256xf8E4M3FN, #partition_aware>
    tt.return %0 : tensor<128x256xf8E4M3FN, #partition_aware>
  }
}

// -----

#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0], [2, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: reduce_16x16
  tt.func @reduce_16x16(%input: tensor<128x128xf32, #mma>) {
    // GFX1250-COUNT-2: rocdl.permlanex16
    %0 = "tt.reduce"(%input) <{axis = 1 : i32}> ({
      ^bb0(%arg1: f32 , %arg2: f32):
      %2 = "arith.maxnumf"(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %2 : f32 }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
   tt.return
  }
}

// -----

// Test lowering of operations with PartitionedSharedEncodingAttr using padded_shared layout
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#inner_padded_piece = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [4, 16]}>
#partitioned_piece = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded_piece}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_alloc
  tt.func @partitioned_shared_padded_local_alloc(%arg0: tensor<16x16xf16, #blocked>) {
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }

  // GFX1250-LABEL: partitioned_shared_padded_multibuffer_index
  tt.func @partitioned_shared_padded_multibuffer_index(%index: i32) {
    %parent = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #partitioned_piece, #smem, mutable>
    // GFX1250: llvm.mlir.constant(128 : i32)
    // GFX1250: llvm.getelementptr
    %view = ttg.memdesc_index %parent[%index] : !ttg.memdesc<2x16x16xf16, #partitioned_piece, #smem, mutable> -> !ttg.memdesc<16x16xf16, #partitioned_piece, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_shared}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_multibuffer_subslice
  tt.func @partitioned_shared_multibuffer_subslice() -> tensor<16x16xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<5x16x16xf16, #partitioned, #smem, mutable>
    // GFX1250: [[STAGE_OFFSET:%.*]] = llvm.mlir.constant(256 : i32)
    // GFX1250-COUNT-2: llvm.getelementptr {{.*}}[[STAGE_OFFSET]]
    %1 = ttg.memdesc_subslice %0 [2, 0, 0] : !ttg.memdesc<5x16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable, 5x16x16>
    %2 = ttg.memdesc_index %1[%c0] : !ttg.memdesc<3x16x16xf16, #partitioned, #smem, mutable, 5x16x16> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    // GFX1250: llvm.load
    %3 = ttg.local_load %2 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %3 : tensor<16x16xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_load
  tt.func @partitioned_shared_padded_local_load() -> tensor<16x16xf16, #blocked> {
    // Allocate and then load from partitioned shared memory
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{[0-9]+}}xf16>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 128 : i32, 65664 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return %1 : tensor<16x16xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: partitioned_shared_padded_local_store
  tt.func @partitioned_shared_padded_local_store(%arg0: tensor<16x16xf16, #blocked>) {
    // Allocate and then store to partitioned shared memory
    // GFX1250: llvm.mlir.addressof @global_smem
    // GFX1250-COUNT-4: llvm.store {{.*}} : vector<{{[0-9]+}}xf16>, !llvm.ptr<3>
    %0 = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32, 128 : i32, 65664 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: @bf16_mulf
  tt.func @bf16_mulf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xf8E4M3FN, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250: rocdl.cvt.scale.pk8.bf16.fp8
    // GFX1250: llvm.fmul {{.*}} : vector<2xbf16>
    %0 = tt.fp_to_fp %arg1 : tensor<64xf8E4M3FN, #blocked> -> tensor<64xbf16, #blocked>
    %1 = arith.mulf %arg0, %0 : tensor<64xbf16, #blocked>
    tt.return %1 : tensor<64xbf16, #blocked>
  }

  // GFX1250-LABEL: @bf16_addf
  tt.func @bf16_addf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250-NOT: llvm.fadd {{.*}} : f32
    // GFX1250: llvm.fadd {{.*}} : vector<2xbf16>
    %0 = arith.addf %arg0, %arg1 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }

  // GFX1250-LABEL: @bf16_subf
  tt.func @bf16_subf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
    // GFX1250-NOT: llvm.fsub {{.*}} : f32
    // GFX1250: llvm.fsub {{.*}} : vector<2xbf16>
    %0 = arith.subf %arg0, %arg1 : tensor<64xbf16, #blocked>
    tt.return %0 : tensor<64xbf16, #blocked>
  }
}

// -----

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The eight-warp helper uses the caller's precomputed offset.
  // GFX1250-LABEL: llvm.func internal @outlined_indices_8
  // GFX1250: [[WAVE:%.*]] = rocdl.wave.id : i32
  // GFX1250: [[OFFSET:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // GFX1250: [[REL:%.*]] = llvm.sub [[WAVE]], [[OFFSET]] : i32
  // GFX1250: [[MASK:%.*]] = llvm.mlir.constant(7 : i32) : i32
  // GFX1250: llvm.and [[REL]], [[MASK]] : i32
  tt.func private @outlined_indices_8() -> tensor<256xi32, #blocked8> attributes {noinline = true, "ttg.num-warps" = 8 : i32, "ttg.warp-id-offset" = 4 : i32} {
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked8>
    tt.return %range : tensor<256xi32, #blocked8>
  }
}
