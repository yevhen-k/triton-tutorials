import json
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(
            model_config, "response:json"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        output_dtype = self.output_dtype
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # Get input
            input_tensor = pb_utils.get_input_tensor_by_name(request, "config:json")
            input_arr: np.ndarray = input_tensor.as_numpy()
            input_json = json.loads(input_arr.tobytes())

            # Got input json: `input_json`.
            # Process `input_json` here
            # [...]

            # Prepare response
            json_string = json.dumps(input_json)
            outputs = np.frombuffer(json_string.encode(), dtype=np.uint8)
            out_tensor = pb_utils.Tensor("response:json", outputs.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

            responses.append(inference_response)

        return responses
