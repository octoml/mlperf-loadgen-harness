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

For e.g.

```
python src/main.py ../models/yolov5s.onnx --runner threadpool --ep CPUExecutionProvider --concurrency 4
```

---
Thelio Benchmarks

YoloV5, CPUEP,  Inline,       0,  32.10 QPS

YoloV5, CPUEP,  ThreadPool,   2,  36.53 QPS
YoloV5, CPUEP,  ThreadPool,   4,  38.40 QPS
YoloV5, CPUEP,  ThreadPool,   8,  35.70 QPS
YoloV5, CPUEP,  ThreadPool,   16, 31.54 QPS

YoloV5, CPUEP,  ProcessPool,  2,  13.14 QPS
YoloV5, CPUEP,  ProcessPool,  4,  24.08 QPS
YoloV5, CPUEP,  ProcessPool,  8,  32.93 QPS
YoloV5, CPUEP,  ProcessPool,  16, 29.77 QPS
