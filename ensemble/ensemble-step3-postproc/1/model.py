import json
from typing import Dict, List, Tuple

import cv2
import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.model_config = model_config = json.loads(args["model_config"])

        # Get INPUT0 configuration
        input_dets = pb_utils.get_input_config_by_name(model_config, "dets:postproc")
        assert isinstance(input_dets, dict)
        # Convert Triton types to numpy types
        self.input_dets_dtype = pb_utils.triton_string_to_numpy(input_dets["data_type"])

        # Get INPUT1 configuration
        input_labels = pb_utils.get_input_config_by_name(
            model_config, "labels:postproc"
        )
        assert isinstance(input_labels, dict)
        # Convert Triton types to numpy types
        self.input_labels_dtype = pb_utils.triton_string_to_numpy(
            input_labels["data_type"]
        )

        # Get OUTPUT0 configuration
        output_scores = pb_utils.get_output_config_by_name(
            model_config, "scores:tensor"
        )
        assert isinstance(output_scores, dict)
        # Convert Triton types to numpy types
        self.output_scores_dtype = pb_utils.triton_string_to_numpy(
            output_scores["data_type"]
        )

        # Get OUTPUT1 configuration
        output_labels = pb_utils.get_output_config_by_name(
            model_config, "labels:tensor"
        )
        assert isinstance(output_labels, dict)
        # Convert Triton types to numpy types
        self.output_labels_dtype = pb_utils.triton_string_to_numpy(
            output_labels["data_type"]
        )

        # Get OUTPUT2 configuration
        output_boxes = pb_utils.get_output_config_by_name(model_config, "boxes:tensor")
        assert isinstance(output_boxes, dict)
        # Convert Triton types to numpy types
        self.output_boxes_dtype = pb_utils.triton_string_to_numpy(
            output_boxes["data_type"]
        )

        # Get model parameters
        self.class_ids: np.ndarray = np.fromstring(
            model_config["parameters"]["class_ids"]["string_value"],
            sep=", ",
            dtype=np.int32,
        )
        self.threshold = float(model_config["parameters"]["threshold"]["string_value"])
        print(f"{self.class_ids=}")
        print(f"{self.threshold=}")

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # Getting input tensors
            dets_tensor = pb_utils.get_input_tensor_by_name(request, "dets:postproc")
            dets: np.ndarray = dets_tensor.as_numpy()

            labels_tensor_in = pb_utils.get_input_tensor_by_name(
                request, "labels:postproc"
            )
            labels_in: np.ndarray = labels_tensor_in.as_numpy()

            # Actual postprocessing
            scores, labels_out, boxes = self.postprocess(dets, labels_in)

            # Preparing output tensors
            scores_tensor = pb_utils.Tensor(
                "scores:tensor", scores.astype(self.output_scores_dtype)
            )
            labels_tensor_out = pb_utils.Tensor(
                "labels:tensor", labels_out.astype(self.output_labels_dtype)
            )
            boxes_tensor = pb_utils.Tensor(
                "boxes:tensor", boxes.astype(self.output_boxes_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    scores_tensor,
                    labels_tensor_out,
                    boxes_tensor,
                ]
            )

            responses.append(inference_response)

        return responses

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def box_cxcywh_to_xyxy_numpy(self, x: np.ndarray) -> np.ndarray:
        x_c, y_c, w, h = np.split(x, 4, axis=-1)
        b = np.concatenate(
            [
                x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None),
                y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None),
                x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None),
                y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None),
            ],
            axis=-1,
        )
        return b

    def postprocess(
        self,
        pred_boxes: np.ndarray,
        pred_logits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get the bounding box and class scores
        # Apply sigmoid activation
        prob = self.sigmoid(pred_logits)

        # Get the top-k values and indices
        flat_prob = prob[0].flatten()
        top_indexes = np.argsort(flat_prob)[::-1]
        top_values = np.take_along_axis(flat_prob, top_indexes, axis=0)
        scores = top_values
        top_boxes = top_indexes // pred_logits.shape[2]
        labels = top_indexes % pred_logits.shape[2]

        # Gather boxes corresponding to top-k indices
        boxes = self.box_cxcywh_to_xyxy_numpy(pred_boxes[0])
        boxes = np.take_along_axis(
            boxes, np.expand_dims(top_boxes, axis=-1).repeat(4, axis=-1), axis=0
        )

        # NOTE: rescaling must be done on the client side

        # Filter detections based on the confidence threshold
        high_confidence_indices = np.argmin(scores > self.threshold)
        scores = scores[:high_confidence_indices]
        labels = labels[:high_confidence_indices]
        boxes = boxes[:high_confidence_indices]

        return scores, labels, boxes
