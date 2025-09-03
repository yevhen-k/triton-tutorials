import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import tritonclient.grpc as grpcclient
import tritonclient.utils as utils
from PIL import Image, ImageDraw

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]


def open_image(path: str) -> np.ndarray:
    # Check if the path is a URL (starts with 'http://' or 'https://')
    if path.startswith("http://") or path.startswith("https://"):
        # If it's a URL, use requests to fetch the image
        # img = Image.open(io.BytesIO(requests.get(path).content))
        img_str = requests.get(path).content
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        # If it's a local file path, open the image directly
        if Path.exists(Path(path)):
            # img = Image.open(path)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")

    return img


def preprocess_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Preprocess the input image for inference."""

    # Resize the image to the model's input size
    # image = image.resize((size[0], size[1]))
    image = cv2.resize(image, (size[0], size[1]))

    # Convert image to numpy array and normalize pixel values
    image = np.array(image).astype(np.float32) / 255.0

    # Normalize
    image = ((image - MEANS) / STDS).astype(np.float32)

    # Change dimensions from HWC to CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def request(
    model_name: str,
    grpc_client: grpcclient.InferenceServerClient,
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Prepare the input and output
    inputs = [
        grpcclient.InferInput(
            "input", image.shape, utils.np_to_triton_dtype(image.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(image)

    outputs = [
        grpcclient.InferRequestedOutput("dets"),
        grpcclient.InferRequestedOutput("labels"),
    ]

    # Perform inference
    results = grpc_client.infer(model_name, inputs, outputs=outputs)

    # Process the output
    pred_boxes = results.as_numpy("dets")
    pred_logits = results.as_numpy("labels")

    return pred_boxes, pred_logits


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def box_cxcywh_to_xyxy_numpy(x):
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


def post_process(
    pred_boxes,
    pred_logits,
    origin_height,
    origin_width,
    confidence_threshold,
    max_number_boxes,
):
    """Post-process the model's output to extract bounding boxes and class information."""
    # Get the bounding box and class scores
    # Apply sigmoid activation
    prob = sigmoid(pred_logits)

    # Get the top-k values and indices
    flat_prob = prob[0].flatten()
    topk_indexes = np.argsort(flat_prob)[-max_number_boxes:][::-1]
    topk_values = np.take_along_axis(flat_prob, topk_indexes, axis=0)
    scores = topk_values
    topk_boxes = topk_indexes // pred_logits.shape[2]
    labels = topk_indexes % pred_logits.shape[2]

    # Gather boxes corresponding to top-k indices
    boxes = box_cxcywh_to_xyxy_numpy(pred_boxes[0])
    boxes = np.take_along_axis(
        boxes, np.expand_dims(topk_boxes, axis=-1).repeat(4, axis=-1), axis=0
    )

    # Rescale box locations
    target_sizes = np.array([[origin_height, origin_width]])
    img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
    scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)
    boxes = boxes * scale_fct[0, :]

    # Filter detections based on the confidence threshold
    high_confidence_indices = np.argmin(scores > confidence_threshold)
    scores = scores[:high_confidence_indices]
    labels = labels[:high_confidence_indices]
    boxes = boxes[:high_confidence_indices]

    return scores, labels, boxes


def save_detections(image_path, boxes, labels, save_image_path):
    """Draw bounding boxes and class labels on the original image."""
    # Load the original image
    # image = open_image(image_path).convert("RGB")
    image = open_image(image_path)
    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    # Loop over the boxes
    for box, label in zip(boxes.astype(int), labels.astype(int)):
        if label in (1, 3):  # Person or Car label id
            # Draw the rectangle (box) on the image
            draw.rectangle(box.tolist(), outline="red", width=4)

    # Save the image with the rectangle and text
    image.save(save_image_path)


def save_predictions(pred_boxes: np.ndarray, pred_logits: np.ndarray) -> None:
    print(f"{pred_boxes.shape=}")
    # pred_boxes.shape=(1, 300, 4)
    with Path.open(Path("assets/pred_boxes.npy"), "wb") as f:
        np.save(f, pred_boxes)

    print(f"{pred_logits.shape=}")
    # pred_logits.shape = (1, 300, 91)
    with Path.open(Path("assets/pred_logits.npy"), "wb") as f:
        np.save(f, pred_logits)


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
    model_name = "ensemble-step2-inference"
    image_path = args.image

    confidence_threshold = 0.5
    max_number_boxes = 100

    image = open_image(image_path)
    origin_width, origin_height = image.shape[1], image.shape[0]

    total_request = 0
    start = time.time()
    image_arr = preprocess_image(image=image, size=size)
    for i in range(count):
        print(f"Progress: {i + 1}/{count}", end="\r")

        start_request = time.time()
        pred_boxes, pred_logits = request(
            model_name=model_name,
            grpc_client=grpc_client,
            image=image_arr,
        )

        end_request = time.time()
        total_request += end_request - start_request

    scores, labels, boxes = post_process(
        pred_boxes,
        pred_logits,
        origin_height,
        origin_width,
        confidence_threshold,
        max_number_boxes,
    )

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

    save_detections(image_path, boxes, labels, "sample.jpg")
    save_predictions(pred_boxes, pred_logits)
