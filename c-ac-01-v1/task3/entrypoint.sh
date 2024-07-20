#!/bin/bash

/opt/websockify/run 5901 --web=/opt/noVNC --wrap-mode=ignore -- vncserver :1 -3dwm &

sleep 5

startxfce4 &

nginx &

jupyter notebook --ip 0.0.0.0 --port 8080 --allow-root --no-browser --NotebookApp.base_url="/lab" --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.default_url="notebooks/Streaming%20and%20Visual Profiling.ipynb"
