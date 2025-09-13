import json
from typing import Dict, List

import cv2
import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])

        # Get INPUT0 configuration
        input_config = pb_utils.get_input_config_by_name(model_config, "in:jpg")
        # Convert Triton types to numpy types
        self.input_dtype = pb_utils.triton_string_to_numpy(input_config["data_type"])

        # Get OUTPUT0 configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "out:jpg")
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(
        self, requests: "List[pb_utils.InferenceRequest]"
    ) -> "List[pb_utils.InferenceResponse]":
        output_dtype = self.output_dtype

        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # Get INPUT0 image
            input_tensor = pb_utils.get_input_tensor_by_name(request, "in:jpg")
            input_arr: np.ndarray = input_tensor.as_numpy()
            image = cv2.imdecode(input_arr, cv2.IMREAD_UNCHANGED)

            # Prepare output
            # outputs = np.frombuffer(image.tobytes(), dtype=np.uint8)
            # out_tensor = pb_utils.Tensor("out:jpg", outputs.astype(output_dtype))
            out_tensor = pb_utils.Tensor("out:jpg", image.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

            responses.append(inference_response)

        return responses
