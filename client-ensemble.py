import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
import tritonclient.utils as utils
from PIL import Image, ImageDraw


def request(
    model_name: str,
    grpc_client: grpcclient.InferenceServerClient,
    np_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # add batch dim
    np_data = np.expand_dims(np_data, axis=[0])

    inputs = [
        grpcclient.InferInput(
            "ensemble:in:jpg", np_data.shape, utils.np_to_triton_dtype(np_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(np_data)

    outputs = [
        grpcclient.InferRequestedOutput("ensemble:scores:tensor"),
        grpcclient.InferRequestedOutput("ensemble:labels:tensor"),
        grpcclient.InferRequestedOutput("ensemble:boxes:tensor"),
    ]

    # Perform inference
    results = grpc_client.infer(model_name, inputs, outputs=outputs)

    # Process the output
    scores = results.as_numpy("ensemble:scores:tensor")
    labels = results.as_numpy("ensemble:labels:tensor")
    boxes = results.as_numpy("ensemble:boxes:tensor")

    return scores, labels, boxes


def open_image(path: str) -> np.ndarray:
    # If it's a local file path, open the image directly
    if Path.exists(Path(path)):
        # img = Image.open(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError(f"The file {path} does not exist.")

    return img


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
    model_name = "ensemble-model"

    image_path = args.image
    image = open_image(image_path)
    origin_width, origin_height = image.shape[1], image.shape[0]
    with Path.open(image_path, "rb") as f:
        data = f.read()

    np_data = np.frombuffer(data, dtype=np.uint8)

    total_request = 0
    start = time.time()
    for i in range(count):
        print(f"Progress: {i + 1}/{count}", end="\r")

        start_request = time.time()
        scores, labels, boxes = request(
            model_name=model_name,
            grpc_client=grpc_client,
            np_data=np_data,
        )

        end_loop = time.time()

        total_request += end_loop - start_request

    end = time.time()
    total_sec = end - start
    fps = count / total_sec
    print(f"Statistics on {count} requests:")

    data = {"Statistic": [], "Value": [], "Unit": []}
    data["Statistic"].append("Total Speed")
    data["Value"].append(f"{fps:.2f}")
    data["Unit"].append("FPS")

    data["Statistic"].append("Request")
    data["Value"].append(f"{total_request / total_sec * 100:.2f}")
    data["Unit"].append("%")

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
