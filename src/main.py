import logging
import os
import typing

import mlperf_loadgen

from loadgen.harness import Harness
from loadgen.runners import (
    ModelRunnerBasic,
    ModelRunnerMultiProcessingPool,
    ModelRunnerProcessPoolExecutor,
    ModelRunnerThreadPoolExecutor,
)
from ort import ORTModel, ORTModelInputSampler

logger = logging.getLogger(__name__)


def main():

    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Offline
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.offline_expected_qps = 20
    settings.min_query_count = 100
    settings.min_duration_ms = 1000 * 10
    # Duration isn't enforced in offline mode
    # Instead, it is used to determine total sample count via
    # target_sample_count = Slack (1.1) * TargetQPS (1) * TargetDuration ()
    # samples_per_query = Max(min_query_count, target_sample_count)

    output_path = "results"
    os.makedirs(output_path, exist_ok=True)

    output_settings = mlperf_loadgen.LogOutputSettings()
    output_settings.outdir = output_path
    output_settings.copy_summary_to_stdout = True

    log_settings = mlperf_loadgen.LogSettings()
    log_settings.log_output = output_settings
    log_settings.enable_trace = False

    model_path = "../yolov5/yolov5s.onnx"
    ep = "CPUExecutionProvider"
    model = ORTModel(model_path, ep)
    model_dataset = ORTModelInputSampler(model)

    runner = ModelRunnerMultiProcessingPool(
        model, max_concurrency=4, granular_tasks=True
    )
    # runner = LoadGenModelRunnerProcessPoolExecutor(model, max_concurrency=4)
    # runner = LoadGenModelRunnerThreadPoolExecutor(model, max_concurrency=4)
    # runner = LoadGenModelRunnerSimple(model)
    harness = Harness(model_dataset, runner)

    try:
        query_sample_libary = mlperf_loadgen.ConstructQSL(
            100,  # Total sample count
            100,  # Num to load in RAM at a time
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

    finally:
        mlperf_loadgen.DestroySUT(system_under_test)
        mlperf_loadgen.DestroyQSL(query_sample_libary)
        logger.info("Test Completed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(threadName)s - %(name)s %(funcName)s: %(message)s",
    )
    main()
