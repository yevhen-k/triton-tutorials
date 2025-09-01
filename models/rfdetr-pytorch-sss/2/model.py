import json
from typing import Dict, List

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Absolute model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # path: str = os.path.join(args["model_repository"], args["model_version"])
        model_repository_path: str = args["model_repository"]

        self.model = RFDETRMedium(
            pretrain_weights=f"{model_repository_path}/rf-detr-medium.pth"
        )
        # self.model.optimize_for_inference(batch_size=32)

        self.model_config = model_config = json.loads(args["model_config"])
        print(f"{model_config=}")

        # Get model parameters
        self.class_ids: np.ndarray = np.fromstring(
            model_config["parameters"]["class_ids"]["string_value"],
            sep=", ",
            dtype=np.int32,
        )
        self.threshold = float(model_config["parameters"]["threshold"]["string_value"])

        # Get INPUT0 configuration
        input_config = pb_utils.get_input_config_by_name(model_config, "in:jpg")
        # Convert Triton types to numpy types
        self.input_dtype = pb_utils.triton_string_to_numpy(input_config["data_type"])

        # Get OUTPUT0 configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "detections:json"
        )
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    async def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        images = []
        for request in requests:
            # Get INPUT0 image
            input_tensor = pb_utils.get_input_tensor_by_name(request, "in:jpg")
            input_arr: np.ndarray = input_tensor.as_numpy()
            image = cv2.imdecode(input_arr, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        detections = self.model.predict(images, threshold=self.threshold)

        if len(images) == 1:  # no batch
            responses = []
            predictions = []
            for class_id, box, confidence in zip(
                detections.class_id, detections.xyxy, detections.confidence
            ):
                # Get box coordinates
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if class_id in self.class_ids:
                    predictions.append(
                        {
                            "polygon": {
                                "tl": [x1, y1],
                                "tr": [x2, y1],
                                "br": [x2, y2],
                                "bl": [x1, y2],
                            },
                            "confidence": float(confidence),
                            "class": COCO_CLASSES[class_id],
                        }
                    )

            # Prepare output
            json_string = json.dumps(predictions)
            outputs = np.frombuffer(json_string.encode(), dtype=np.uint8)
            out_tensor = pb_utils.Tensor(
                "detections:json", outputs.astype(self.output_dtype)
            )
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

            responses.append(inference_response)

            return responses

        else:
            print(f">>> Batch size: {len(images)}")
            responses = []
            for detecion in detections:
                predictions = []
                for class_id, box, confidence in zip(
                    detecion.class_id, detecion.xyxy, detecion.confidence
                ):
                    # Get box coordinates
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if class_id in self.class_ids:
                        predictions.append(
                            {
                                "polygon": {
                                    "tl": [x1, y1],
                                    "tr": [x2, y1],
                                    "br": [x2, y2],
                                    "bl": [x1, y2],
                                },
                                "confidence": float(confidence),
                                "class": COCO_CLASSES[class_id],
                            }
                        )
                # Prepare output
                json_string = json.dumps(predictions)
                outputs = np.frombuffer(json_string.encode(), dtype=np.uint8)
                out_tensor = pb_utils.Tensor(
                    "detections:json", outputs.astype(self.output_dtype)
                )

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor]
                )

                responses.append(inference_response)

            return responses
