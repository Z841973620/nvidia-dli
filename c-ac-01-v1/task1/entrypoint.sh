#!/bin/bash

jupyter notebook --ip 0.0.0.0 --port 80 --allow-root --no-browser --NotebookApp.base_url="/" --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.default_url="notebooks/AC_CUDA_C.ipynb"
