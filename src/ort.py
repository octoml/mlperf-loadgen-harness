import typing

import numpy as np
import onnx
import onnxruntime as ort

from loadgen.model import Model, ModelInput, ModelInputSampler

ONNX_TO_NP_TYPE_MAP = {
    "tensor(bool)": np.bool,
    "tensor(int)": np.int32,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(uint64)": np.uint64,
    "tensor(int64)": np.int64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(string)": np.string_,
}


class ORTModel(Model):
    def __init__(self, model_path, ep="CPUExecutionProvider"):
        model = onnx.load(model_path)
        session_options = ort.SessionOptions()
        # session_options.execution_mode
        # session_options.intra_op_num_threads = 0
        # session_options.inter_op_num_threads = 0
        session_eps = [ep]
        self.session = ort.InferenceSession(
            model.SerializeToString(), session_options, providers=session_eps
        )

    def predict(self, input: ModelInput):
        return self.session.run(None, input)


class ORTModelInputSampler(ModelInputSampler):
    def __init__(self, model: ORTModel):
        input_defs = model.session.get_inputs()
        self.inputs: typing.Dict[str, typing.Tuple[np.dtype, typing.List[int]]] = dict()
        for input in input_defs:
            input_name = input.name
            input_type = ONNX_TO_NP_TYPE_MAP[input.type]
            input_dim = [
                1 if (x is None or (type(x) is str)) else x for x in input.shape
            ]
            self.inputs[input_name] = (input_type, input_dim)

    def sample(self, id_: int) -> ModelInput:
        input = dict()
        for name, spec in self.inputs.items():
            val = np.random.random_sample(spec[1]).astype(spec[0])
            input[name] = val
        return input