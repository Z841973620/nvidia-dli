FROM nvcr.io/nvidia/nvhpc:23.9-devel-cuda12.2-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt update && apt install -y --no-install-recommends python3 python3-pip && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.20 && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab==3.4.8 matplotlib scipy==1.9.2 numpy==1.20 notebook==6.4.13 && rm -rf ~/.cache/pip

WORKDIR /dli/task

ADD entrypoint.sh /usr/local/bin/
COPY jupyter/ /root/.jupyter/
COPY bashrc /root/.bashrc
COPY dli/  /dli/
