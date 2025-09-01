FROM nvcr.io/nvidia/tritonserver:24.08-py3

WORKDIR /app-server

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache python3 -m pip install -r requirements.txt

CMD ["bash"] 
