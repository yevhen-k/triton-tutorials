import argparse
import asyncio
import json
import time
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import tritonclient.grpc.aio as grpcclient
import tritonclient.utils as utils


async def request(
    model_name: str,
    grpc_client: grpcclient.InferenceServerClient,
    np_data: np.ndarray,
    semaphore: asyncio.Semaphore,
) -> np.ndarray:
    np_data = np.expand_dims(np_data, axis=0)

    inputs = [
        grpcclient.InferInput(
            "in:jpg", np_data.shape, utils.np_to_triton_dtype(np_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(np_data)

    outputs = [
        grpcclient.InferRequestedOutput("detections:json"),
    ]

    async with semaphore:
        results = await grpc_client.infer(
            model_name,
            inputs,
            outputs=outputs,
            model_version="2",
        )

    response_np_arr = results.as_numpy("detections:json")
    # print(results.get_response(as_json=True))
    # print(response_np_arr.shape)

    return response_np_arr


def postprocess(image_path: str, detections: List[Dict]) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    color_green = (0, 255, 0)
    for detection in detections:
        tl = detection["polygon"]["tl"]
        br = detection["polygon"]["br"]
        cv2.rectangle(image, tl, br, color_green, 3)

    cv2.imwrite("sample.jpg", image)


async def main(args: argparse.Namespace):
    benchmark = args.benchmark
    count = 1
    if benchmark:
        count = 1000

    grpc_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
    model_name = "rfdetr-pytorch-sss"

    image_path = "assets/bus.jpg"
    with open(image_path, "rb") as f:
        data = f.read()

    np_data = np.frombuffer(data, dtype=np.uint8)

    total_request = 0
    semaphore = asyncio.Semaphore(32)
    start = time.time()
    # NOTE: use of asyncio.create_task()
    # tasks = []
    # for i in range(count):
    #     print(f"Progress: {i+1}/{count}", end="\r")

    #     task = asyncio.create_task(
    #         request(
    #             model_name=model_name,
    #             grpc_client=grpc_client,
    #             np_data=np_data,
    #             semaphore=semaphore,
    #         )
    #     )
    #     tasks.append(task)

    # # Await all tasks to complete
    # response_np_arrays = await asyncio.gather(*tasks, return_exceptions=True)
    # response_np_arr = response_np_arrays[0]

    async with asyncio.TaskGroup() as tg:
        response_np_arrays = [
            tg.create_task(
                request(
                    model_name=model_name,
                    grpc_client=grpc_client,
                    np_data=np_data,
                    semaphore=semaphore,
                )
            )
            for _ in range(count)
        ]
    response_np_arrays = [arr.result() for arr in response_np_arrays]
    response_np_arr = response_np_arrays[0]
    end_loop = time.time()
    total_request += end_loop - start

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

    resp_json = json.loads(response_np_arr.tobytes())

    postprocess(image_path, resp_json)


if __name__ == "__main__":
    # assert False, "UNIMPLEMENTED: needs to implement async on the triton server side"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        help="If prvided, run statistics collection on 1000 requests",
        action="store_true",
    )
    args = parser.parse_args()
    asyncio.run(main(args=args))
