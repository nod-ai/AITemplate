from typing import List

from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects import tensor

from ..base import IntImm, Tensor

def get_mlir_dtype(dtype: str):
    match dtype:
        case "float16":
            return F16Type.get()
        case "float32":
            return F32Type.get()
        case "float":
            return F32Type.get()
        case "int":
            return I32Type.get()
        case "int32":
            return I32Type.get()
        case "int64":
            return I64Type.get()
        case _:
            return None

def is_int_type(dtype: str):
    return dtype == "int" or dtype == "int32" or dtype == "int64"

def is_float_type(dtype: str):
    return dtype == "float" or dtype == "float16" or dtype == "float32"

def get_tensor_type(tensor: Tensor):
    dtype = get_mlir_dtype(tensor._attrs["dtype"])

    shape_attr_list = tensor._attrs["shape"]
    shape = [a.value() if isinstance(a, IntImm) else -1 for a in shape_attr_list]

    return RankedTensorType.get(shape, dtype)

def get_function_signature(sorted_graph: List[Tensor]):
    input_types = []
    output_types = []

    input_tensors = []
    output_tensors = []
    for tensor in sorted_graph:
        if tensor._attrs["is_input"]:
            tensor_type = get_tensor_type(tensor)
            tensor._attrs["arg_index"] = len(input_types)
            input_types.append(tensor_type)
            input_tensors.append(tensor)

        if tensor._attrs["is_output"]:
            tensor_type = get_tensor_type(tensor)
            tensor._attrs["return_index"] = len(output_types)
            output_types.append(tensor_type)

    assert len(output_types) == 1, "Require a single output"

    return FunctionType.get(inputs=input_types, results=output_types), input_tensors, output_tensors

def get_converted_output(output: Tensor):
    if not "mlir_value" in output._attrs:
        out_tensor_type = get_tensor_type(output)
        return tensor.EmptyOp(out_tensor_type.shape, out_tensor_type.element_type).result
    return output._attrs["mlir_value"]

def get_dummy_splat_tensor(tensor: Tensor, val):
    tensor_type = get_tensor_type(tensor)
    dtype = tensor._attrs["dtype"]
    if is_int_type(dtype):
        element = IntegerAttr.get(tensor_type.element_type, val)
    elif is_float_type(dtype):
        element = FloatAttr.get(tensor_type.element_type, val)
    else:
        return None
    dense_resource = DenseElementsAttr.get_splat(tensor_type, element)
    return arith.ConstantOp(tensor_type, dense_resource).result
