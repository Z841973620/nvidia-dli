FROM nvcr.io/nvidia/cuda:12.2.2-base-ubuntu20.04

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

RUN apt-get update -y && apt-get install -y --no-install-recommends python3-pip cuda-minimal-build-12-2 && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple notebook==6.5.6 && rm -rf ~/.cache/pip

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib numba==0.57.1 nvidia-ml-py3 scipy==1.9.2 numpy==1.21 && rm -rf ~/.cache/pip

ENV PYTHONPATH=/dli/assessment/notebook_helpers

WORKDIR /dli/task

COPY dli/  /dli/
ADD entrypoint.sh /usr/local/bin/
