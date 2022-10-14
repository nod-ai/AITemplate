from mlir.dialects import linalg
from mlir.dialects import tensor
from mlir.dialects import arith

from ..utils import get_tensor_type, get_dummy_splat_tensor

def gemm_rcr_bias_fast_gelu_conversion_pattern(op, inputs):
    output_tensor = op._attrs["outputs"][0]
    return get_dummy_splat_tensor(output_tensor, 3.0)

def gemm_rcr_bias_conversion_pattern(op, inputs):
    output_tensor = op._attrs["outputs"][0]
    return get_dummy_splat_tensor(output_tensor, 4.0)
