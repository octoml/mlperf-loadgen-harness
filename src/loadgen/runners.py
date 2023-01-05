import abc
import concurrent.futures
import logging
import multiprocessing
import threading
import typing
import os
import functools

# import triton

from loadgen.harness import ModelRunner, QueryCallback, QueryInput
from loadgen.model import Model, ModelFactory, ModelInput

logger = logging.getLogger(__name__)

######## Runner implementations


class ModelRunnerInline(ModelRunner):
    def __init__(self, model_factory: ModelFactory):
        self.model = model_factory.create()

    def issue_query(self, queries: QueryInput, callback: QueryCallback):
        for query_id, model_input in queries.items():
            output = self.model.predict(model_input)
            callback({query_id: output})


class ModelRunnerTriton(ModelRunner):
    def __init__(self, model_factory: ModelFactory):
        self.model_name = os.path.splitext(os.path.basename(model_factory.model_path))[
            0
        ]
        self.executor = None
        self.session = None
        # self.executor = triton.Executor(
        #     "/home/giqbal/Source/triton/developer_tools/server/examples/models",
        #     "/home/giqbal/Source/triton/developer_tools/server/examples/backends",
        # )

        # self.session = self.executor.create_session(self.model_name)

    def issue_query(self, queries: QueryInput, callback: QueryCallback):
        # predict_cb = functools.partial(ModelRunnerTriton._triton_callback, callback)
        predict_cb = functools.partial(ModelRunnerTriton._triton_callback_id, callback)
        for query_id, model_input in queries.items():
            # print(query_id)

            # output = self.session.predict(model_input)
            # callback({query_id: output})

            # self.session.predict_with_callback(
            #     model_input,
            #     predict_cb,
            #     query_id,
            # )

            self.session.predict_with_callback_queue(
                model_input,
                predict_cb,
                query_id,
            )

    @staticmethod
    def _triton_callback(query_cb: QueryCallback, query_id, query_result):
        query_cb({query_id: None})

    @staticmethod
    def _triton_callback_id(query_cb: QueryCallback, query_id):
        query_cb({query_id: None})


class ModelRunnerPoolExecutor(ModelRunner):
    def __init__(self, max_concurrency: int):
        self.max_concurrency = max_concurrency
        self.futures: typing.Dict[concurrent.futures.Future, int] = {}
        self.executor: typing.Optional[concurrent.futures.Executor] = None
        self.callback_fn: typing.Optional[QueryCallback] = None

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.executor:
            self.executor.shutdown(True)
        return super().__exit__(_exc_type, _exc_value, _traceback)

    def issue_query(self, queries: QueryInput, callback: QueryCallback):
        assert not self.futures
        self.callback_fn = callback
        predictor_fn = self.get_predictor()
        for query_id, model_input in queries.items():
            f = self.executor.submit(predictor_fn, model_input)
            self.futures[f] = query_id
            f.add_done_callback(self._future_callback)

    @abc.abstractmethod
    def get_predictor(self) -> typing.Callable[[ModelInput], typing.Any]:
        pass

    def _future_callback(self, f: concurrent.futures.Future):
        query_id = self.futures.pop(f)
        query_result = f.result()
        self.callback_fn({query_id: query_result})


class ModelRunnerThreadPoolExecutor(ModelRunnerPoolExecutor):
    def __init__(self, model_factory: ModelFactory, max_concurrency: int):
        super().__init__(max_concurrency)
        self.model = model_factory.create()

    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrency, thread_name_prefix="LoadGen"
        )
        return self

    def get_predictor(self) -> typing.Callable[[ModelInput], typing.Any]:
        return self.model.predict


class ModelRunnerThreadPoolExecutorWithTLS(ModelRunnerPoolExecutor):
    tls: threading.local

    def __init__(self, model_factory: ModelFactory, max_concurrency: int):
        super().__init__(max_concurrency)
        self.model_factory = model_factory

    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrency,
            thread_name_prefix="LoadGen",
            initializer=ModelRunnerThreadPoolExecutorWithTLS._tls_init,
            initargs=(self.model_factory,),
        )
        return self

    def get_predictor(self) -> typing.Callable[[ModelInput], typing.Any]:
        return ModelRunnerThreadPoolExecutorWithTLS._tls_predict

    @staticmethod
    def _tls_init(model_factory: ModelFactory):
        ModelRunnerThreadPoolExecutorWithTLS.tls = threading.local()
        ModelRunnerThreadPoolExecutorWithTLS.tls.model = model_factory.create()

    @staticmethod
    def _tls_predict(input: ModelInput):
        return ModelRunnerThreadPoolExecutorWithTLS.tls.model.predict(input)


class ModelRunnerProcessPoolExecutor(ModelRunnerPoolExecutor):
    _model: Model

    def __init__(self, model_factory: ModelFactory, max_concurrency: int):
        super().__init__(max_concurrency)
        ModelRunnerProcessPoolExecutor._model = model_factory.create()

    def __enter__(self):
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_concurrency
        )
        return self

    def get_predictor(self) -> typing.Callable[[ModelInput], typing.Any]:
        return ModelRunnerProcessPoolExecutor._predict

    @staticmethod
    def _predict(input: ModelInput):
        result = ModelRunnerProcessPoolExecutor._model.predict(input)
        return result


class ModelRunnerMultiProcessingPool(ModelRunner):
    _model: Model

    def __init__(
        self,
        model_factory: ModelFactory,
        max_concurrency: int,
    ):
        self.max_concurrency = max_concurrency
        self.task: typing.Optional[multiprocessing.pool.MapResult] = None
        self.callback_fn: typing.Optional[QueryCallback] = None
        ModelRunnerMultiProcessingPool._model = model_factory.create()

    def __enter__(self):
        self.pool = multiprocessing.Pool(self.max_concurrency)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.pool:
            self.pool.terminate()
        return super().__exit__(_exc_type, _exc_value, _traceback)

    def issue_query(self, queries: QueryInput, callback: QueryCallback):
        assert self.task is None
        assert self.callback_fn is None
        inputs = [[query_id, model_input] for query_id, model_input in queries.items()]
        self.callback_fn = callback
        self.task = self.pool.starmap_async(
            ModelRunnerMultiProcessingPool._predict_with_id, inputs
        )

    def flush_queries(self):
        task_result = self.task.get()
        result = {query_id: query_result for query_id, query_result in task_result}
        self.callback_fn(result)
        self.task = None
        self.callback_fn = None

    @staticmethod
    def _predict_with_id(query_id: int, input: ModelInput):
        result = ModelRunnerMultiProcessingPool._model.predict(input)
        return (query_id, result)
