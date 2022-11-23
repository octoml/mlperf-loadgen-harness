A simple harness to run an ONNX model in various concurrency and replication settings against MLCommon's [LoadGen](https://github.com/mlcommons/inference/tree/master/loadgen) to measure throughput.

## Setup

Install requirement.txt and requirements.dev.txt if developing in VSCode.
You may need to build the LoadGen wheel if the provided version (Linux/x64) doesn't match your setup. 
It can be built via instructions [here](https://github.com/mlcommons/inference/tree/master/loadgen/demos/lon#setup)
```
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
```

## Usage

```
usage: main.py [-h] [-o OUTPUT] [-r {inline,threadpool,threadpool+replication,processpool,processpool+mp}] [--concurrency CONCURRENCY] [--ep EP] [--intraop INTRAOP]
               [--interop INTEROP] [--execmode {sequential,parallel}]
               model_path

positional arguments:
  model_path            path to input model

optional arguments:
  -h, --help            show this help message and exit
  -o, --output OUTPUT
                        path to store loadgen results
  -r, --runner {inline,threadpool,threadpool+replication,processpool,processpool+mp}
                        model runner
  --concurrency CONCURRENCY
                        concurrency count for runner
  --ep EP               Execution Provider
  --intraop INTRAOP     IntraOp threads
  --interop INTEROP     InterOp threads
  --execmode {sequential,parallel}
                        Execution Mode
 ```

