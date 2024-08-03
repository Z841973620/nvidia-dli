#!/bin/bash

/opt/websockify/run 5901 --web=/opt/noVNC --wrap-mode=ignore -- vncserver :1 -3dwm &

sleep 5

startxfce4 &

nginx &

jupyter lab --ip 0.0.0.0 --port 8080 --allow-root --no-browser --NotebookApp.base_url="/lab" --NotebookApp.token="" --NotebookApp.password=""
