A simple harness to run an ONNX model in various concurrency and replication settings against MLCommon's [LoadGen](https://github.com/mlcommons/inference/tree/master/loadgen) to measure throughput.

## Setup

Install requirement.txt in your Python 3.8 (virtual) environment.
Also install requirements.dev.txt if developing in VSCode.

You may need to build the LoadGen wheel if the provided versions in the /bin folder (Linux x64 and aarch64) do not match your setup.
If necessary, it can be built via instructions [here](https://github.com/mlcommons/inference/tree/master/loadgen/demos/lon#setup)
```
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
```

## Usage

The following arguments are supported.

```
usage: main.py  [-h] [-o OUTPUT] 
                [--ep EP] [--execmode {sequential,parallel}]
                [--intraopthreads INTRAOPTHREADS..] [--interopthreads INTEROPTHREADS..] 
                [--runner {inline,threadpool,threadpoolmultiinstance,processpool,ray,batchedthreadpool,batchedprocesspool}...] 
                [--concurrency CONCURRENCY...] 
                [--model_input_dims x=1,y=2]
                model_path

positional arguments:
  model_path                        path to input model

optional arguments:
  -h, --help                        show this help message and exit
  -o, --output OUTPUT               path to store loadgen results
  -r, --runner {inline,threadpool,threadpool+replication,processpool,processpool+mp}
                                    model runner
  --concurrency CONCURRENCY         concurrency count for runner
  --ep EP                           Execution Provider
  --execmode {sequential,parallel}  Execution Mode (Default Sequential)
  --intraop_threads INTRAOP         IntraOp threads
  --interop_threads INTEROP         InterOp threads
  --model_input_dims                Specific values for any dynamic input axes
 ```

For e.g.

```
python src/main.py models/yolov5s.onnx --ep CPUExecutionProvider --runner threadpool --concurrency 4
python src/main.py models/yolov5s.onnx --ep CPUExecutionProvider --runner batchedthreadpool batchedprocesspool --concurrency 1 2 4 8 16 24 32 --intraopthreads 0 1 2 4 8 16 24 32  --interopthreads 0 1 2
python src/main.py models/hf-distilbert-uncased.onnx --model_input_dims sequence=256,batch=2 --ep CPUExecutionProvider --runner inline
```
