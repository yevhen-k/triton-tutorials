import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
import tritonclient.utils as utils


def request(
    model_name: str,
    grpc_client: grpcclient.InferenceServerClient,
    np_data: np.ndarray,
) -> np.ndarray:
    # add batch dim
    np_data = np.expand_dims(np_data, axis=[0])

    inputs = [
        grpcclient.InferInput(
            "in:jpg", np_data.shape, utils.np_to_triton_dtype(np_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(np_data)

    outputs = [
        grpcclient.InferRequestedOutput("out:tensor"),
    ]

    results = grpc_client.infer(model_name, inputs, outputs=outputs)

    response_np_arr = results.as_numpy("out:tensor")

    return response_np_arr


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
    model_name = "ensemble-step1-preproc"

    with Path.open(args.image, "rb") as f:
        data = f.read()

    np_data = np.frombuffer(data, dtype=np.uint8)

    total_request = 0
    start = time.time()
    for i in range(count):
        print(f"Progress: {i + 1}/{count}", end="\r")

        start_request = time.time()
        response_np_arr = request(
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

    # Revert preprocessing:
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    # response shape CHW  [3, 576, 576]
    # image    shape HWC [ 576, 576, 3 ]
    print(f"{response_np_arr.shape=}")
    image = np.transpose(response_np_arr[0], (1, 2, 0))
    image = image * stds + means
    image = image * 255.0

    ok = cv2.imwrite("sample.jpg", image)
    assert ok
