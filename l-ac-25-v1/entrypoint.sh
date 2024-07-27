#!/bin/bash

/usr/local/bin/dockerd &

chown labuser.labuser --recursive /dli/task

exec su labuser -c 'jupyter notebook --ip 0.0.0.0 --port 80 --allow-root --no-browser --NotebookApp.base_url="" --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.default_url="notebooks/Building%20Container%20Images.ipynb"'
