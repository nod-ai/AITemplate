# Converting an AIT model to MLIR

Support for converting an AIT model to the Linalg dialect in MLIR is demoed here.

## Building MLIR Python Bindings
To enable this conversion you need to build LLVM with MLIR python bindings enabled. A detailed explanation can be found [here](https://mlir.llvm.org/docs/Bindings/Python/). First, you need to clone LLVM.

```bash
git clone https://github.com/llvm/llvm-project
cd llvm-project
```
(Latest verified commit: 9363071303ec59bc9e0d9b989f08390b37e3f5e4)

Install the MLIR python dependencies, typically in a virtual environment.
```bash
python -m pip install -r mlir/python/requirements.txt
```

LLVM can be built on Linux with the following cmake command. See [this](https://mlir.llvm.org/getting_started/) for Windowssupport.
```bash
mkdir build && cd build
cmake -G Ninja ../llvm
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python)
```

Then export the `PYTHONPATH` variable to make the built python bindings available.
```bash
export PYTHONPATH=$(cd build && pwd)/tools/mlir/python_packages/mlir_core
```

## Running the sample
The sample can be run with the following
```bash
python convert_to_mlir_example.py
```
