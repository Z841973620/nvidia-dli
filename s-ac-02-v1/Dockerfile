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

RUN apt-get update -y && apt-get install -y --no-install-recommends cuda-nsight-systems-12-2 && apt clean && rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/opt/VirtualGL/bin:/opt/TurboVNC/bin:/opt/docker
ENV DISPLAY=:1
ENV NVIDIA_DRIVER_CAPABILITIES=compat32,compute,display,graphics,utility,video
ENV PYTHONPATH=/dli/assessment/notebook_helpers

ADD *.tar /opt/
ADD *.deb /tmp/

RUN cd /tmp && apt update -y && apt install -y --no-install-recommends python3-pip python python-numpy libglfw3-dev nginx build-essential gcc make pkg-config libxau-dev libxdmcp-dev libxcb1-dev ca-certificates curl libc6-dev libglu1 libsm6 libxv1 x11-xkb-utils xauth xfonts-base xkb-data vim libgl1-mesa-glx dbus-x11 xubuntu-desktop xubuntu-icon-theme xfce4-terminal xorg dbus-x11 x11-xserver-utils libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxkbcommon-x11-0 apt-transport-https msttcorefonts zip unzip && dpkg -i *.deb && rm -f /tmp/*.deb && apt clean && rm -rf /var/lib/apt/lists/*
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && echo '/usr/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && ldconfig && echo '/usr/lib/x86_64-linux-gnu/libGL.so.1' >> /etc/ld.so.preload && echo '/usr/lib/x86_64-linux-gnu/libEGL.so.1' >> /etc/ld.so.preload && sed -i 's/$host:/unix:/g' /opt/TurboVNC/bin/vncserver && cd /opt/websockify && make && mkdir -p /root/.vnc && echo "nvidia" | vncpasswd -f > /root/.vnc/passwd && chmod 0600 /root/.vnc/passwd && rm /etc/nginx/nginx.conf && rm /usr/share/backgrounds/xfce/*

ADD nvidia-wallpaper.jpg /usr/share/backgrounds/xfce/xfce-stripes.png
ADD nginx.conf /etc/nginx/

WORKDIR /dli/task

RUN pip3 uninstall notebook -y && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab==3.4.8 matplotlib scipy==1.9.2 numpy==1.20 && rm -rf ~/.cache/pip

COPY jupyter/ /root/.jupyter/
COPY bashrc /root/.bashrc
COPY dli/  /dli/

RUN mkdir /root/Desktop && ln -sT /dli/task /root/Desktop/reports

ADD entrypoint.sh /usr/local/bin/
