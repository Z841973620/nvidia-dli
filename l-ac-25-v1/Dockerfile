FROM ubuntu:20.04

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,compat32,display,graphics,utility,video

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ADD docker.tgz /usr/local/bin/

RUN apt update && apt install -y --no-install-recommends fuse-overlayfs squashfs-tools ca-certificates curl gnupg iptables && curl https://repo.download.nvidia.cn/baseos/ubuntu/focal/dgx-repo-files.tgz | tar xzf - -C / && apt update && apt install -y --no-install-recommends nvidia-container-toolkit && apt clean && rm -rf /var/lib/apt/lists/*

ADD *.xz /opt/

RUN apt update && apt install -y --no-install-recommends sudo python3-pip libssl-dev cryptsetup pkg-config uuid-dev libseccomp-dev build-essential && apt clean && rm -rf /var/lib/apt/lists/* && cd /opt && export PATH="${PATH}:/opt/go/bin" && cd /opt/singularity && ./mconfig && cd ./builddir && make && make install && apt autoremove -y libssl-dev cryptsetup pkg-config uuid-dev libseccomp-dev build-essential && cd ~ && rm -rf /opt/*

COPY localtime /etc/

RUN useradd -s /bin/bash -m labuser && passwd -d -u labuser && groupadd docker && gpasswd -a labuser docker && echo "labuser ALL=(ALL:ALL) /usr/local/bin/docker" >> /etc/sudoers && echo "labuser ALL=(ALL:ALL) /usr/local/bin/singularity" >> /etc/sudoers
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple notebook==6.4.13 numpy==1.19.1 hpccm==20.8.0 netCDF4==1.5.4 matplotlib==3.3.1 && rm -rf ~/.cache/pip

WORKDIR /dli/task

COPY dli/ /dli/
ADD entrypoint.sh /usr/local/bin/
