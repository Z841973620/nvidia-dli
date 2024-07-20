FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /dli/task

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab==3.4.8 dgl==1.1.3 matplotlib==3.5.3 ogb==1.3.6 && rm -rf ~/.cache/pip
RUN mkdir ~/.dgl && echo '{"backend": "pytorch"}' > ~/.dgl/config.json

COPY dli/  /dli/

ADD entrypoint.sh /usr/local/bin/
