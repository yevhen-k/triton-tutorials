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
        assert isinstance(input_config, dict)
        # Convert Triton types to numpy types
        self.input_dtype = pb_utils.triton_string_to_numpy(input_config["data_type"])

        # Get OUTPUT0 configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "out:tensor")
        assert isinstance(output_config, dict)
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self.output_shape = output_config["dims"]

        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    async def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        responses = []
        # for loop for batch requests (disabled in our case)
        print(f"bs = {len(requests)}", flush=True)
        for request in requests:
            # Get INPUT0 image
            input_tensor = pb_utils.get_input_tensor_by_name(request, "in:jpg")
            input_arr: np.ndarray = input_tensor.as_numpy()
            image = cv2.imdecode(input_arr, cv2.IMREAD_UNCHANGED)

            # Prepare output
            # [x] reshape image
            # [x] normalize image
            # [x] change dimensions from HWC to CHW
            # [ ] add batch dimension (NOTE: should we?)

            image = cv2.resize(image, (self.output_shape[1], self.output_shape[2]))
            # image = cv2.resize(image, (self.output_shape[2], self.output_shape[3]))
            image = np.array(image).astype(np.float32) / 255.0
            image = ((image - self.means) / self.stds).astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)

            out_tensor = pb_utils.Tensor("out:tensor", image.astype(self.output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

            responses.append(inference_response)

        return responses
