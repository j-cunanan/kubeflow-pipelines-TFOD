FROM nvcr.io/nvidia/tensorflow:20.11-tf2-py3
# Use tensorflow/tensorflow:2.3.1-gpu for machines with <=CUDA10

RUN apt-get update && apt-get install -y \
    wget \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    libgl1-mesa-glx

WORKDIR /pipelines
COPY models /pipelines/models
ENV PYTHONPATH=$PYTHONPATH:/pipelines/models

RUN (cd /pipelines/models/research && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /pipelines/models/research

RUN python -m pip install -U pip
RUN python -m pip install pyyaml
RUN pip install numpy==1.18.2
RUN python -m pip install ./

WORKDIR /pipelines/models/research/object_detection
