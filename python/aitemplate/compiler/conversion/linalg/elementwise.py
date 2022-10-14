from mlir.ir import *

from mlir.dialects import linalg
from mlir.dialects import tensor
from mlir.dialects import arith

from ..utils import get_tensor_type, get_dummy_splat_tensor, get_converted_output, is_mlir_int_type, is_mlir_float_type
from ...ops.common.epilogue import FuncEnum

def insert_elementwise_operation(elementwise_func, arg0, arg1):
    match elementwise_func:
        case FuncEnum.ADD:
            if is_mlir_int_type(arg0.type):
                return arith.AddIOp(arg0, arg1)
            elif is_mlir_float_type(arg0.type):
                return arith.AddFOp(arg0, arg1)
        case _:
            return None

def elementwise_conversion_pattern(op, inputs):
    output_tensor = op._attrs["outputs"][0]
    output_type = get_tensor_type(output_tensor)
    output_rank = output_type.rank

    iterator_types_attr = ArrayAttr.get([StringAttr.get("parallel")] * output_rank)
    indexing_maps = []
    for input in inputs:
        input_type = RankedTensorType(input.type)
        input_rank = input_type.rank
        exprs = []
        for index, dim in enumerate(input_type.shape):
            if dim == 1:
                exprs.append(AffineConstantExpr.get(0))
            else:
                exprs.append(AffineDimExpr.get(index + output_rank - input_rank))

        indexing_maps.append(AffineMap.get(output_rank, 0, exprs))
    indexing_maps.append(AffineMap.get_identity(output_rank))
    indexing_maps_attr = ArrayAttr.get(
        [AffineMapAttr.get(am) for am in indexing_maps])

    converted_output = get_converted_output(output_tensor)

    generic_op = linalg.GenericOp(
        result_tensors=[output_type],
        inputs=inputs,
        outputs=[converted_output],
        indexing_maps=indexing_maps_attr,
        iterator_types=iterator_types_attr,
        doc=None,
        library_call=None)

    block_arg_types = [RankedTensorType(input.type).element_type for input in inputs]
    block_arg_types += [RankedTensorType(converted_output.type).element_type]
    block = generic_op.regions[0].blocks.append(*block_arg_types)
    with InsertionPoint(block):
        generic_args = block.arguments
        value = insert_elementwise_operation(op._attrs["func"], generic_args[0], generic_args[1])
        linalg.YieldOp([value])

    return generic_op.result
