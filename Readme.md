
## Notes

`pb_utils.get_input_tensor_by_name(request, "some_name")` takes `InferenceRequest` instance and returns `Tensor` .

- `get_input_tensor_by_name`: [triton_python_backend_utils.py#L123](https://github.com/triton-inference-server/python_backend/blob/8b5a055e6f2cdd22cf6d3644e5822b01ffc62ce1/src/resources/triton_python_backend_utils.py#L123)
- `InferenceRequest`: [_request.py#L49](https://github.com/triton-inference-server/core/blob/70b908ca74b27407ae7c33b26ef26401d50aa871/python/tritonserver/_api/_request.py#L49)
- `Tensor`: [_tensor.py#L64](https://github.com/triton-inference-server/core/blob/main/python/tritonserver/_api/_tensor.py#L64)


## Run Triton

```bash
docker run --gpus=1 --rm --net=host -v ${PWD}/models:/models nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
# OR
docker run --gpus=1 --rm --net=host -v ${PWD}/models:/models yevhenk10s/triton-pytorch-rfdetr:24.08-py3 tritonserver --model-repository=/models
```

## Build Docker Image

```bash
docker build --rm -t triton-pytorch-rfdetr .
docker tag triton-pytorch-rfdetr yevhenk10s/triton-pytorch-rfdetr:24.08-py3
docker push yevhenk10s/triton-pytorch-rfdetr:24.08-py3
```

## References
- Triton Ensemble Model for deploying Transformers into production: https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3
- Serving Multiple Models with NVIDIA Triton Inference Server: https://apxml.com/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/serving-models-triton-server
- Executing Multiple Models with Model Ensembles: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_5-Model_Ensembles/README.html
- Triton Inference Server + TensorRT + metrics: https://github.com/Koldim2001/Triton_example
- trtexec CLI: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html
-  Optimizing Model Deployments with Triton Model Analyzer: https://www.youtube.com/watch?v=UU9Rh00yZMY
-  Custom params in `config.pbtxt`: https://github.com/triton-inference-server/common/blob/2e41435a59f7fe1f5f73df5355ae7433a15a4650/protobuf/model_config.proto#L1669
- Async inference requests: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/bls.html

## Supported Hardware by TensorRT
- https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- https://developer.nvidia.com/cuda-gpus
- https://developer.nvidia.com/cuda-legacy-gpus

```bash
/opt/cuda/extras/demo_suite/deviceQuery
```
Shows:
```
Device 0:                                       "NVIDIA GeForce GTX 1050 Ti"
CUDA Capability Major/Minor version number:     6.1
```

But TensorRT supports at least `7.5` compute capability.