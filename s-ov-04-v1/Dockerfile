FROM nvcr.io/nvidia/modulus/modulus:22.07

RUN pip install ipywidgets gdown

ADD dli /dli
ADD entrypoint.sh /usr/local/bin/

WORKDIR /dli/task

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
