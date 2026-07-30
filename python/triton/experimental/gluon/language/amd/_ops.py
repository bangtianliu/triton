import math

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import builtin, _unwrap_if_constexpr
from .._layouts import DotOperandLayout
from ._layouts import AMDWMMALayout


@builtin
def optimize_register_pressure(_semantic=None):
    """Request the low-spill register-pressure policy for this kernel.

    This is a semantic optimization request rather than the name of a
    particular LLVM pass. The AMD backend remains free to implement it using
    the target's scheduling and rematerialization mechanisms.
    """
    _semantic.builder.set_register_pressure_policy("minimize-spills")


@builtin
def rematerialized_range(start, end, layout, _semantic=None):
    """Place a distributed integer range at this source location.

    This has the same values and type as ``gl.arange(start, end, layout)``.
    It tells the backend that recomputing lane/warp coordinates here is
    preferable to carrying the range through a long software pipeline. It
    does not prescribe physical registers or alter numerical semantics.
    """
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    layout = _unwrap_if_constexpr(layout)
    _check(isinstance(start, int) and not isinstance(start, bool), lambda: "start must be a constexpr integer")
    _check(
        isinstance(end, int) and not isinstance(end, bool) and end > start,
        lambda: "end must be a constexpr integer greater than start")
    _check(layout is not None, lambda: "layout must be explicit")
    result_type = ttgl.distributed_type(ttgl.int32, [end - start], layout)
    handle = _semantic.builder.create_rematerialized_range(result_type.to_ir(_semantic.builder), start, end)
    return ttgl.tensor(handle, result_type)


@builtin
def commit_mfma(value, preserve, _semantic=None):
    """Commit one or more native vector MFMA results and thread a resident operand.

    ``value`` may be one tensor or a tuple of independent native fragments.
    Each fragment must be a direct transient ``scheduled_mfma`` result.
    A single value returns ``(value, preserve)``; a tuple is flattened to
    ``(*values, preserve)`` because Gluon builtins return flat SSA tuples.
    ``preserve`` has no numerical role in ``value``; consume its returned copy
    in the next source stage so residency and ordering across the completion
    boundary are explicit SSA dependencies. CDNA4 lowering derives native
    fragment widths and the result-hazard delay from operand layouts.
    """
    single_value = isinstance(value, ttgl.tensor)
    values = (value,) if single_value else value
    _check(
        isinstance(values, (tuple, ttgl.tuple)) and len(values) > 0
        and all(isinstance(item, ttgl.tensor) for item in values)
        and isinstance(preserve, ttgl.tensor),
        lambda: "value must be a tensor or nonempty tensor tuple and preserve must be a tensor")
    inputs = tuple(values) + (preserve,)
    handles = _semantic.builder.create_mfma_commit(
        [item.handle for item in inputs]
    )
    outputs = tuple(
        ttgl.tensor(handle, item.type) for handle, item in zip(handles, inputs)
    )
    if single_value:
        return outputs[0], outputs[-1]
    return outputs


@builtin
def scheduled_mfma(a, b, acc, resident_operand=None, accumulator="persistent", initialize=False, _semantic=None):
    """Update independent native fragments with source-controlled scheduling.

    The per-wave fragments of ``a`` and ``b`` form a Cartesian product over
    the output grid. Each output fragment remains an independent accumulator
    chain; instructions are emitted in N-major, M-minor, K-reduction order.

    ``resident_operand`` may be 0 or 1 when one input is intentionally carried
    through a software pipeline. ``accumulator`` selects ``"transient"`` for
    a phase-local result or ``"persistent"`` for an accumulator carried across
    phases. These are lifetime roles, not storage classes, physical register
    numbers, or tuple widths; lowering chooses target storage and derives
    native tuples from the Gluon layouts.

    When ``initialize=True``, ``acc`` defines only result shape and layout and
    the native accumulators start from zero. A transient result may be the
    accumulator of one following transient ``scheduled_mfma``; the terminal
    result must be passed directly to ``commit_mfma``. This expresses both the
    dependency chain and its completion/live-through boundary.

    All active lanes of a wave must execute the operation uniformly. The
    pinned MLIR ``LLVM::InlineAsmOp`` has no convergent-call attribute, so this
    precondition is part of the primitive contract until lowering can use an
    explicitly convergent target operation.
    """
    resident_operand = _unwrap_if_constexpr(resident_operand)
    accumulator = _unwrap_if_constexpr(accumulator)
    initialize = _unwrap_if_constexpr(initialize)
    _check(isinstance(a, ttgl.tensor) and isinstance(b, ttgl.tensor), lambda: "a and b must be distributed tensors")
    _check(isinstance(acc, ttgl.tensor), lambda: "acc must be a distributed tensor")
    _check(
        resident_operand is None or
        (isinstance(resident_operand, int) and not isinstance(resident_operand, bool) and resident_operand in {0, 1}),
        lambda: "resident_operand must be None, 0, or 1",
    )
    _check(
        accumulator in {"transient", "persistent"},
        lambda: 'accumulator must be either "transient" or "persistent"',
    )
    _check(isinstance(initialize, bool), lambda: "initialize must be a constexpr bool")
    resident_role = {
        None: "none",
        0: "lhs",
        1: "rhs",
    }[resident_operand]
    handle = _semantic.builder.create_scheduled_mfma(
        acc.type.to_ir(_semantic.builder),
        a.handle,
        b.handle,
        acc.handle,
        resident_role,
        accumulator,
        initialize,
    )
    return ttgl.tensor(handle, acc.type)


def _wrap_scaled_upcast_result(handle, elem_type, semantic):
    shape = semantic.builder.get_shape_from_tensor(handle)
    layout = semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(elem_type, shape, layout)
    return ttgl.tensor(handle, ret_ty)


def _verify_wmma(version, a, b, acc):
    _check(acc is not None, lambda: "acc is required")

    layout = acc.type.layout
    _check(
        isinstance(layout, AMDWMMALayout) and layout.version == version,
        lambda: f"Expected layout to be an instance of AMDWMMALayout with version {version}")

    a_layout = a.type.layout
    _check(
        isinstance(a_layout, DotOperandLayout) and isinstance(a_layout.parent, AMDWMMALayout)
        and a_layout.parent.version == version,
        lambda: "Expected a's layout to be a DotOperandLayout with parent matching AMDWMMALayout")

    b_layout = b.type.layout
    _check(
        isinstance(b_layout, DotOperandLayout) and isinstance(b_layout.parent, AMDWMMALayout)
        and b_layout.parent.version == version,
        lambda: "Expected b's layout to be a DotOperandLayout with parent matching AMDWMMALayout")


def _wmma(version, a, b, acc, semantic):
    """ Shared implementation for AMD WMMA operations for Gluon builtins """
    _verify_wmma(version, a, b, acc)

    handle = semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                          out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)


def _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, scale_fn, semantic):
    """ Shared implementation for AMD WMMA scaled and MFMA scaled operation. """

    def _get_scale_shape(op_idx, operand, format, scale_factor):
        operand_shape = [s for s in operand.type.shape]
        scale_shape = operand_shape
        unpack_factor = 2 if format == "e2m1" else 1
        if op_idx == 0:
            k = scale_shape[-1] * unpack_factor
            scale_shape[-1] = k // scale_factor
        else:
            k = scale_shape[-2] * unpack_factor
            scale_shape[-2] = k // scale_factor
            scale_shape[-2], scale_shape[-1] = scale_shape[-1], scale_shape[-2]
        return scale_shape

    def _get_default_scale_dtype_and_unit_value(op_idx):
        default_value_by_dtype = {ttgl.uint8: 0x7F, ttgl.float8e4nv: 1.0}

        if a_scale is None and b_scale is None:
            return ttgl.uint8, 0x7F

        if a_format == b_format == "e2m1":
            # Fp4 x Fp4 requries to use the same scale dtype for both operands.
            other_scale = b_scale if op_idx == 0 else a_scale
            return other_scale.dtype, default_value_by_dtype[other_scale.dtype]

        return ttgl.uint8, 0x7F

    def _create_and_broadcast_default_scale(op_idx, scale, format, scale_factor):
        operand = a if op_idx == 0 else b

        scale_shape = _get_scale_shape(op_idx, operand, format, scale_factor)
        if isinstance(scale, ttgl.tensor) and scale.numel.value != 1:
            # In the case of scale pre-shuffling, the input shape is different from the default shape. We only check
            # the number of elements here.
            assert math.prod(scale_shape) == scale.numel.value, "Incompatible scale shape"
            return scale

        scale_layout = scale_fn(operand.type.layout, scale_shape, scale_factor)
        scale_value = _unwrap_if_constexpr(scale)
        if scale_value is None:
            scale_dtype, scale_value = _get_default_scale_dtype_and_unit_value(op_idx)
        elif isinstance(scale_value, int):
            scale_dtype = ttgl.uint8
        elif isinstance(scale_value, float):
            scale_dtype = ttgl.float8e4nv
        else:
            scale_dtype = scale.dtype

        return semantic.full(scale_shape, scale_value, scale_dtype, scale_layout)

    scale_factor = semantic.deduce_scale_factor(a, a_scale, a_format, True, b, b_scale, b_format, True)

    a_scale = _create_and_broadcast_default_scale(0, a_scale, a_format, scale_factor)
    b_scale = _create_and_broadcast_default_scale(1, b_scale, b_format, scale_factor)
    output = semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                 rhs_k_pack=True, out_dtype=ttgl.float32)
    return ttgl.tensor(output.handle, acc.type)


def _scaled_upcast(src, scale, elem_type, axis, semantic):
    _check(isinstance(src.type, ttgl.distributed_type),
           lambda: f"Expected src to have a distributed_type but got {src.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16},
           lambda: f"Expected elem_type to be fp16 or bf16 but got {elem_type}")

    if src.dtype in {ttgl.float8e4nv, ttgl.float8e5}:
        _check(axis is None, lambda: "axis must be None for fp8 scaled_upcast")
        _check(scale.type.shape == src.type.shape,
               lambda: f"Expected scale shape for fp8 scaled_upcast to be {src.type.shape} but got {scale.type.shape}")
        _check(
            scale.type.layout == src.type.layout,
            lambda: f"Expected scale layout for fp8 scaled_upcast to be {src.type.layout} but got {scale.type.layout}")
        # Note: bf16 is allowed due to CDNA3/CDNA4 conversion before passing to scaled_upcast
        _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
               lambda: f"Unsupported scale dtype for fp8 scaled_upcast: {scale.dtype}")
        ret_ty = scale.type.with_element_ty(elem_type)
        handle = semantic.builder.create_scaled_upcast_fp8(ret_ty.to_ir(semantic.builder), src.handle, scale.handle)
        return _wrap_scaled_upcast_result(handle, elem_type, semantic)

    _check(src.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected packed fp4 input in int8/uint8 or fp8 input, but got {src.dtype}")
    _check(axis is not None, lambda: "axis is required for packed fp4 scaled_upcast")

    rank = len(src.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    expected_shape = list(src.type.shape)
    expected_shape[axis] *= 2
    _check(
        scale.type.shape[:axis] + scale.type.shape[axis + 1:] == expected_shape[:axis] + expected_shape[axis + 1:],
        lambda: f"Expected scale shape for scaled_upcast to match output shape on non-axis dims: "
        f"{expected_shape}, but got {scale.type.shape}")
    _check(
        scale.type.shape[axis] > 0 and expected_shape[axis] % scale.type.shape[axis] == 0,
        lambda: f"Expected output axis extent {expected_shape[axis]} to be divisible by scale axis extent "
        f"{scale.type.shape[axis]}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
           lambda: f"Unsupported scale dtype for fp4 scaled_upcast: {scale.dtype}")

    handle = semantic.builder.create_scaled_upcast_fp4(src.handle, scale.handle, elem_type.to_ir(semantic.builder),
                                                       axis)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)
