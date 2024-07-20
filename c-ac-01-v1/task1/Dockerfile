FROM nvcr.io/nvidia/cuda:12.2.2-base-ubuntu20.04

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

RUN echo "deb https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && apt-get update -y && apt-get install -y --no-install-recommends cuda-nvcc-12-2 gcc python3-pip && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple notebook==6.5.6 && rm -rf ~/.cache/pip

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64
ENV PATH=$PATH:/usr/local/cuda-12.2/bin:/usr/local/cuda-12.2/include

WORKDIR /dli/task

COPY dli/  /dli/
ADD entrypoint.sh /usr/local/bin/
