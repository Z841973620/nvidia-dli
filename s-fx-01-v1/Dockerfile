FROM tensorflow/tensorflow:2.13.0-gpu

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONPATH=/dli/assessment/notebook_helpers

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /dli/task

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab==3.4.8 jupyter-client==7.4.8 matplotlib==3.5.3 pandas==1.4.2 numpy==1.22.4 scipy==1.9.3 && rm -rf ~/.cache/pip

COPY dli/  /dli/

ADD entrypoint.sh /usr/local/bin/
