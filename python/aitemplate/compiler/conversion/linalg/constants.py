from mlir.dialects import arith

from ...base import Tensor, Operator

from ..utils import get_tensor_type, get_dummy_splat_tensor

def convert_weight_tensor(tensor: Tensor):
    arith_constant = get_dummy_splat_tensor(tensor, 1.0)
    tensor._attrs["mlir_value"] = arith_constant

def get_converted_inputs(op: Operator):
    converted_inputs = []
    for input in op._attrs["inputs"]:

        if not "mlir_value" in input._attrs:
            convert_weight_tensor(input)

        converted_inputs.append(input._attrs["mlir_value"])
    return converted_inputs
