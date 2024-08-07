FROM nvcr.io/nvidia/deepstream:6.0-triton

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:1

USER root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

RUN echo "deb https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && apt-key adv --fetch-keys http://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && apt update && curl -fkLO https://launchpadlibrarian.net/469281109/ffmpeg_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281110/libavcodec-extra58_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281111/libavcodec58_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281112/libavdevice58_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281113/libavfilter-extra7_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281114/libavfilter7_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281115/libavformat58_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281116/libavresample4_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281117/libavutil56_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281118/libpostproc55_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281119/libswresample3_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/469281120/libswscale5_4.2.2-1ubuntu1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/396685658/libx264-155_0.155.2917+git0a84d98-2_amd64.deb && curl -fkLO https://launchpadlibrarian.net/470282555/libx265-179_3.2.1-1build1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/346760121/libmpeg3-2_1.8.dfsg-2.1_amd64.deb && curl -fkLO https://launchpadlibrarian.net/537882100/libwebm1_1.0.0.28-1_amd64.deb && dpkg -i *.deb ; apt --fix-broken install -y && apt clean && rm -rf /var/lib/apt/lists/* *.deb
RUN curl -fkLO https://192.168.1.11/pypi/pyds/pyds-1.1.0-cp38-cp38-linux_x86_64.whl && apt update && apt install -y --no-install-recommends python3-dev && ln  -s /usr/local/bin/pip3.8 /usr/bin/pip3.8 && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple *.whl jupyterlab==3.4.8 numpy==1.21.2 notebook==6.4.13 pandas matplotlib && rm -rf ~/.cache/pip *.whl && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /dli/task

ADD entrypoint.sh /usr/local/bin/
COPY jupyter/ /root/.jupyter/
COPY bashrc /root/.bashrc
COPY dli/ /dli/

ENV PYTHONPATH=/dli/task
