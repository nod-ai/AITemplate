from typing import List

from mlir.dialects import func
from mlir.ir import *

from ...utils.graph_utils import get_sorted_ops
from ..base import Tensor

from .utils import get_function_signature

# Conversion patterns
from .linalg.elementwise import elementwise_conversion_pattern
from .linalg.gemm import (
    gemm_rcr_bias_fast_gelu_conversion_pattern,
    gemm_rcr_bias_conversion_pattern,
)
from .linalg.layernorm import layernorm_conversion_pattern
from .linalg.constants import get_converted_inputs

# Semantics for a conversion pattern are as follows:
#   Inputs:
#       op, AIT type Operator that is being converted
#       inputs, Operator inputs as MLIR values
#   Outputs:
#       output, result of the Operator as an MLIR value

def get_linalg_conversion_patterns():
    return {
            'elementwise': elementwise_conversion_pattern,
            'gemm_rcr_bias_fast_gelu': gemm_rcr_bias_fast_gelu_conversion_pattern,
            'gemm_rcr_bias': gemm_rcr_bias_conversion_pattern,
            'layernorm': layernorm_conversion_pattern,
    }

def convert_to_linalg(sorted_graph: List[Tensor]):
    #print(sorted_graph)
    sorted_ops = get_sorted_ops(sorted_graph)

    with Context() as context, Location.unknown():
        module = Module.create()

        with InsertionPoint(module.body):
            function_type, inputs, results = get_function_signature(sorted_graph)
            func_op = func.FuncOp(name="forward", type=function_type)

            linalg_conversion_patterns = get_linalg_conversion_patterns()
            with InsertionPoint(func_op.add_entry_block()):

                # Map the mlir function arguments to the graph inputs
                func_args = func_op.entry_block.arguments
                for input, func_arg in zip(inputs, func_args):
                    input._attrs["mlir_value"] = func_arg

                # Lower op by op
                for op in sorted_ops:

                    # Get input values
                    converted_inputs = get_converted_inputs(op)

                    assert len(op._attrs["outputs"]) == 1, "Do single output everything for now"

                    conversion_pattern = linalg_conversion_patterns[op._attrs["op"]]
                    converted_output = conversion_pattern(op, converted_inputs)

                    output_tensor = op._attrs["outputs"][0]
                    output_tensor._attrs["mlir_value"] = converted_output

                # Insert the return statement
                for tensor in sorted_graph:
                    if tensor._attrs["is_output"]:
                        func.ReturnOp([tensor._attrs["mlir_value"]])

    print(module)
