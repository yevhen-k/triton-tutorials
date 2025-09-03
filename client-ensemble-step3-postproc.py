import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
import tritonclient.utils as utils
from PIL import Image, ImageDraw

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]


def open_image(path: str) -> np.ndarray:
    # If it's a local file path, open the image directly
    if Path.exists(Path(path)):
        # img = Image.open(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError(f"The file {path} does not exist.")

    return img


def request(
    model_name: str,
    grpc_client: grpcclient.InferenceServerClient,
    pred_boxes: np.ndarray,
    pred_logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prepare the input and output
    inputs = [
        grpcclient.InferInput(
            "dets:postproc",
            pred_boxes.shape,
            utils.np_to_triton_dtype(pred_boxes.dtype),
        ),
        grpcclient.InferInput(
            "labels:postproc",
            pred_logits.shape,
            utils.np_to_triton_dtype(pred_logits.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(pred_boxes)
    inputs[1].set_data_from_numpy(pred_logits)

    outputs = [
        grpcclient.InferRequestedOutput("scores:tensor"),
        grpcclient.InferRequestedOutput("labels:tensor"),
        grpcclient.InferRequestedOutput("boxes:tensor"),
    ]

    # Perform inference
    results = grpc_client.infer(model_name, inputs, outputs=outputs)

    # Process the output
    scores = results.as_numpy("scores:tensor")
    labels = results.as_numpy("labels:tensor")
    boxes = results.as_numpy("boxes:tensor")

    return scores, labels, boxes


def save_detections(
    image_path: str,
    boxes: np.ndarray,
    origin_height: int,
    origin_width: int,
    save_image_path: str,
) -> None:
    """Draw bounding boxes and class labels on the original image."""
    # Load the original image
    # image = open_image(image_path).convert("RGB")
    image = open_image(image_path)
    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    # Loop over the boxes
    for box in boxes:
        # Rescale box locations
        target_sizes = np.array([[origin_height, origin_width]])
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)
        box = box * scale_fct[0, :]
        # Draw the rectangle (box) on the image
        draw.rectangle(box.tolist(), outline="red", width=4)

    # Save the image with the rectangle and text
    image.save(save_image_path)


def load_detections(dets_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(dets_path), np.load(labels_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        help="If prvided, run statistics collection on 1000 requests",
        action="store_true",
    )
    parser.add_argument(
        "--image",
        help="Path or URL of an image to be processed. For example, `assets/bus.jpg` or `https://www.ultralytics.com/images/bus.jpg`",
        default="assets/bus.jpg",
    )

    args = parser.parse_args()

    benchmark = args.benchmark
    count = 1
    if benchmark:
        count = 1000

    grpc_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
    size = (576, 576)
    model_name = "ensemble-step3-postproc"
    image_path = args.image

    confidence_threshold = 0.5
    max_number_boxes = 100

    image = open_image(image_path)
    origin_width, origin_height = image.shape[1], image.shape[0]

    pred_boxes, pred_logits = load_detections(
        "assets/pred_boxes.npy", "assets/pred_logits.npy"
    )
    print(f"Loaded {pred_boxes.shape=}")
    print(f"Loaded {pred_logits.shape=}")

    total_request = 0
    start = time.time()
    for i in range(count):
        print(f"Progress: {i + 1}/{count}", end="\r")

        start_request = time.time()
        scores, labels, boxes = request(
            model_name=model_name,
            grpc_client=grpc_client,
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
        )

        end_request = time.time()
        total_request += end_request - start_request

    end = time.time()
    fps = count / total_request
    print(f"Statistics on {count} requests:")

    data = {"Statistic": [], "Value": [], "Unit": []}
    data["Statistic"].append("Total Speed")
    data["Value"].append(f"{fps:.2f}")
    data["Unit"].append("FPS")

    data["Statistic"].append("Request Time")
    data["Value"].append(f"{total_request / count * 1000:.2f}")
    data["Unit"].append("ms")

    df = pd.DataFrame(data=data)
    print(df.to_markdown(index=False))

    save_detections(
        image_path=image_path,
        boxes=boxes,
        origin_width=origin_width,
        origin_height=origin_height,
        save_image_path="sample.jpg",
    )
