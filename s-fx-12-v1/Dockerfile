FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /dli/task

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab==4.2.0 ipywidgets==8.1.2 accelerate==0.30.0 auto_gptq==0.7.1 optimum==1.19.2 transformers==4.33.2 && rm -rf ~/.cache/pip

ADD entrypoint.sh /usr/local/bin/

COPY dli/  /dli/
