import concurrent.futures
import logging
import multiprocessing
import typing

from loadgen.harness import ModelRunner, QueryInput, QueryResult
from loadgen.model import Model, ModelInput

logger = logging.getLogger(__name__)

######## Runner implementations


class ModelRunnerBasic(ModelRunner):
    def issue_query(self, queries: QueryInput) -> typing.Optional[QueryResult]:
        result = dict()
        for query_id, query_input in queries.items():
            output = self.model.predict(query_input)
            result[query_id] = output
        return result


class ModelRunnerPoolExecutor(ModelRunner):
    def __init__(self, model: Model, executor: concurrent.futures.Executor):
        super().__init__(model)
        self.executor = executor
        self.futures = None

    def issue_query(self, queries: QueryInput) -> typing.Optional[QueryResult]:
        self.futures = dict()
        for query_id, query_input in queries.items():
            f = self.executor.submit(self.model.predict, query_input)
            self.futures[f] = query_id
        return None

    def flush_queries(self) -> typing.Optional[QueryResult]:
        result = dict()
        for future in concurrent.futures.as_completed(self.futures.keys()):
            query_id = self.futures[future]
            query_result = future.result()
            result[query_id] = query_result
        return result


class ModelRunnerThreadPoolExecutor(ModelRunnerPoolExecutor):
    def __init__(self, model: Model, max_concurrency: int):
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix="LoadGen"
        )
        super().__init__(model, executor)


class ModelRunnerProcessPoolExecutor(ModelRunnerPoolExecutor):
    _model: Model

    def __init__(self, model: Model, max_concurrency: int):
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency)
        ModelRunnerProcessPoolExecutor._model = model
        super().__init__(model, executor)

    def issue_query(self, queries: QueryInput) -> typing.Optional[QueryResult]:
        self.futures = dict()
        for query_id, query_input in queries.items():
            f = self.executor.submit(
                ModelRunnerProcessPoolExecutor._predict, query_input
            )
            self.futures[f] = query_id
        return None

    @staticmethod
    def _predict(input: ModelInput):
        result = ModelRunnerProcessPoolExecutor._model.predict(input)
        return result


class ModelRunnerMultiProcessingPool(ModelRunner):
    _model: Model

    def __init__(
        self, model: Model, max_concurrency: int, granular_tasks: bool = False
    ):
        super().__init__(model)
        ModelRunnerMultiProcessingPool._model = model
        self.pool = multiprocessing.Pool(max_concurrency)
        if granular_tasks:
            self.tasks: typing.List[multiprocessing.ApplyResult] = {}
        else:
            self.task: multiprocessing.ApplyResult = None

    def issue_query(self, queries: QueryInput) -> typing.Optional[QueryResult]:
        if hasattr(self, "tasks"):
            assert len(self.tasks) == 0
            for query_id, query_input in queries.items():
                task = self.pool.apply_async(
                    ModelRunnerMultiProcessingPool._predict, (query_input,)
                )
                self.tasks[task] = query_id
        else:
            assert self.task is None
            inputs = [
                [query_id, query_input] for query_id, query_input in queries.items()
            ]
            self.task = self.pool.starmap_async(
                ModelRunnerMultiProcessingPool._predict_with_id, inputs
            )
            return None

    def flush_queries(self) -> typing.Optional[QueryResult]:
        if hasattr(self, "tasks"):
            result = dict()
            for task, query_id in self.tasks.items():
                task_result = task.get()
                result[query_id] = task_result
            return result
        else:
            task_result = self.task.get()
            result = {query_id: query_result for query_id, query_result in task_result}
            return result

    @staticmethod
    def _predict(input: ModelInput):
        result = ModelRunnerMultiProcessingPool._model.predict(input)
        return result

    @staticmethod
    def _predict_with_id(query_id: int, input: ModelInput):
        result = ModelRunnerMultiProcessingPool._model.predict(input)
        return (query_id, result)
