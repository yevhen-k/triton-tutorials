import json
from typing import Dict, List

import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(
            model_config, "output:tensor"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self.output_shape = output_config["dims"]

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # Get input
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input:tensor")
            input_arr: np.ndarray = input_tensor.as_numpy()

            # Got input numpy array: `input_arr`.
            # Process `input_arr` here
            # [...]
            # print(f"Got np.array: {input_arr}", flush=True)

            # Prepare response
            outputs = input_arr
            out_tensor = pb_utils.Tensor(
                "output:tensor", outputs.astype(self.output_dtype)
            )

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

            responses.append(inference_response)

        return responses
