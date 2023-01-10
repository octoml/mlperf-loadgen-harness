import argparse
import contextlib
import csv
import dataclasses
import itertools
import logging
import os
import re
import typing
from datetime import date, datetime

import mlperf_loadgen
import psutil

from loadgen.harness import Harness, ModelRunner
from loadgen.runners import (
    ModelRunnerBatchedProcessPool,
    ModelRunnerBatchedThreadPool,
    ModelRunnerInline,
    ModelRunnerProcessPoolExecutor,
    ModelRunnerRay,
    ModelRunnerThreadPoolExecutor,
    ModelRunnerThreadPoolMultiInstanceExecutor,
)
from ort import ORTModelFactory, ORTModelInputSampler

logger = logging.getLogger(__name__)


LOADGEN_EXPECTED_QPS = 50
LOADGEN_SAMPLE_COUNT = 100
LOADGEN_DURATION_SEC = 10


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    timestamp: datetime
    execution_provider: str
    execution_mode: str
    runner_name: str
    runner_concurrency: int
    intraop_threads: int
    interop_threads: int
    qps: float
    mean_latency_ms: float


def benchmark(
    model_path: str,
    output_path: str,
    execution_provider: str,
    execution_mode: str,
    runner_name: str,
    runner_concurrency: int,
    intraop_threads: int,
    interop_threads: int,
) -> BenchmarkResult:

    model_factory = ORTModelFactory(
        model_path,
        execution_provider,
        execution_mode,
        intraop_threads,
        interop_threads,
    )

    model_dataset = ORTModelInputSampler(model_factory)

    runner: ModelRunner = None
    if runner_name == "inline":
        runner = ModelRunnerInline(model_factory)
        runner_concurrency = 1
    elif runner_name == "threadpool":
        runner = ModelRunnerThreadPoolExecutor(
            model_factory, max_concurrency=runner_concurrency
        )
    elif runner_name == "threadpool+multiinstance":
        runner = ModelRunnerThreadPoolMultiInstanceExecutor(
            model_factory, max_concurrency=runner_concurrency
        )
    elif runner_name == "processpool":
        runner = ModelRunnerProcessPoolExecutor(
            model_factory, max_concurrency=runner_concurrency
        )
    elif runner_name == "ray":
        runner = ModelRunnerRay(model_factory, max_concurrency=runner_concurrency)
    elif runner_name == "batchedthreadpool":
        runner = ModelRunnerBatchedThreadPool(
            model_factory, max_concurrency=runner_concurrency
        )
    elif runner_name == "batchedprocesspool":
        runner = ModelRunnerBatchedProcessPool(
            model_factory, max_concurrency=runner_concurrency
        )
    else:
        raise ValueError(f"Invalid runner {runner}")

    logger.info(
        f"Benchmark Starting: Model: {model_path}, {execution_provider}, {execution_mode}"
    )
    logger.info(
        f"- Parameters: {runner_name}, Concurrency: {runner_concurrency}, IntraOp: {intraop_threads}, InterOp: {interop_threads}"
    )

    settings = mlperf_loadgen.TestSettings()
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly

    settings.scenario = mlperf_loadgen.TestScenario.Offline
    settings.offline_expected_qps = LOADGEN_EXPECTED_QPS

    settings.min_query_count = LOADGEN_SAMPLE_COUNT * 2
    settings.min_duration_ms = LOADGEN_DURATION_SEC * 1000
    # Duration isn't enforced in offline mode
    # Instead, it is used to determine total sample count via
    # target_sample_count = Slack (1.1) * TargetQPS (1) * TargetDuration ()
    # samples_per_query = Max(min_query_count, target_sample_count)

    model_name = os.path.basename(model_path)
    qualified_output_path = os.path.join(
        output_path,
        model_name,
        runner_name,
        f"concurrency{runner_concurrency}_threads{intraop_threads}x{interop_threads}",
    )
    os.makedirs(qualified_output_path, exist_ok=True)

    output_settings = mlperf_loadgen.LogOutputSettings()
    output_settings.outdir = qualified_output_path
    output_settings.copy_summary_to_stdout = False

    log_settings = mlperf_loadgen.LogSettings()
    log_settings.log_output = output_settings
    log_settings.enable_trace = False

    logger.info(f"- Writing results to {qualified_output_path}")

    with contextlib.ExitStack() as stack:
        stack.enter_context(runner)
        harness = Harness(model_dataset, runner)
        try:
            query_sample_libary = mlperf_loadgen.ConstructQSL(
                LOADGEN_SAMPLE_COUNT,  # Total sample count
                LOADGEN_SAMPLE_COUNT,  # Num to load in RAM at a time
                harness.load_query_samples,
                harness.unload_query_samples,
            )
            system_under_test = mlperf_loadgen.ConstructSUT(
                harness.issue_query, harness.flush_queries
            )

            logger.info("Test Started")
            mlperf_loadgen.StartTestWithLogSettings(
                system_under_test, query_sample_libary, settings, log_settings
            )

            # Parse output file
            output_summary = {}
            output_summary_path = os.path.join(
                qualified_output_path, "mlperf_log_summary.txt"
            )
            with open(output_summary_path, "r") as output_summary_file:
                for line in output_summary_file:
                    m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
                    if m:
                        output_summary[m.group(1).strip()] = m.group(2).strip()

            qps = float(output_summary.get("Samples per second"))
            mean_latency_ms = float(output_summary.get("Mean latency (ns)")) / 1e6
            result_valid = output_summary.get("Result is")

            logger.info(f"- Observed QPS: {qps:0.2f}")
            logger.info(f"- Mean Latency (ms): {mean_latency_ms:0.2f}")
            logger.info(f"- Result: {result_valid}")

            result = BenchmarkResult(
                timestamp=datetime.now(),
                execution_provider=execution_provider,
                execution_mode=execution_mode,
                runner_name=runner_name,
                runner_concurrency=runner_concurrency,
                intraop_threads=intraop_threads,
                interop_threads=interop_threads,
                qps=qps,
                mean_latency_ms=mean_latency_ms,
            )
            return result

        finally:
            mlperf_loadgen.DestroySUT(system_under_test)
            mlperf_loadgen.DestroyQSL(query_sample_libary)
            logger.info(f"Benchmark Completed: {qualified_output_path}")


def main(
    model_path: str,
    output_path: typing.Optional[str],
    execution_provider: str,
    execution_mode: str,
    runner_list: typing.Sequence[str],
    runner_concurrency_list: typing.Sequence[int],
    intraop_threads_list: typing.Sequence[int],
    interop_threads_list: typing.Sequence[int],
):
    output_path = output_path if output_path else "results"
    output_csv_path = os.path.join(
        output_path, os.path.basename(model_path), f"{date.today()}.csv"
    )
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "a") as output_csv_file:
        csv_fields = [f.name for f in dataclasses.fields(BenchmarkResult)]
        csv_writer = csv.DictWriter(output_csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        for (
            runner_name,
            runner_concurrency,
            intraop_threads,
            interop_threads,
        ) in itertools.product(
            runner_list,
            runner_concurrency_list,
            intraop_threads_list,
            interop_threads_list,
        ):
            result = benchmark(
                model_path,
                output_path,
                execution_provider=execution_provider,
                execution_mode=execution_mode,
                runner_name=runner_name,
                runner_concurrency=runner_concurrency,
                intraop_threads=intraop_threads,
                interop_threads=interop_threads,
            )
            csv_row = dataclasses.asdict(result)
            logger.info(f"Benchmark Result: {csv_row}")
            csv_writer.writerow(csv_row)
            output_csv_file.flush()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(threadName)s - %(name)s %(funcName)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", help="path to input model", default="models/yolov5s.onnx"
    )
    parser.add_argument("-o", "--output", help="path to store loadgen results")
    parser.add_argument(
        "--ep", help="Execution Provider", default="CPUExecutionProvider"
    )
    parser.add_argument(
        "--execmode",
        help="Execution Mode",
        choices=["sequential", "parallel"],
        default="sequential",
    )
    parser.add_argument(
        "-r",
        "--runner",
        help="model runner",
        choices=[
            "inline",
            "threadpool",
            "threadpoolmultiinstance",
            "processpool",
            "ray",
            "batchedthreadpool",
            "batchedprocesspool",
        ],
        default="inline",
        nargs="+",
    )

    parser.add_argument(
        "--concurrency",
        help="concurrency count for runner",
        default=[psutil.cpu_count(False)],
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--intraopthreads", help="IntraOp threads", default=[0], type=int, nargs="+"
    )
    parser.add_argument(
        "--interopthreads", help="InterOp threads", default=[0], type=int, nargs="+"
    )

    args = parser.parse_args()
    main(
        args.model_path,
        args.output,
        args.ep,
        args.execmode,
        args.runner,
        args.concurrency,
        args.intraopthreads,
        args.interopthreads,
    )
