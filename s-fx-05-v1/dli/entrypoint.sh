#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
###############################################################################

# JUPYTER_TOKEN will empty in development, but set for
# deployments. It will be applied from the environment
# automatically when running `docker-compose up`.
# For more details see `docker-compose.production.yml`
# and `docker-compose.override.yml`

# (1) This example script is used as the entrypoint for our Docker container.
# This is the place to activate a conda environment if you are using one.

source /root/.bashrc

#pip3 install geopandas scipy veroviz

python -m ipykernel install --user --name=reopt

jupyter lab \
        --ip 0.0.0.0                               `# Bind to all network interfaces` \
        --allow-root                               `# Allow running as the root user` \
        --no-browser                               `# Do not attempt to launch a browser` \
        --NotebookApp.base_url="/lab"              `# Set a base URL for the lab` \
        --NotebookApp.token="$JUPYTER_TOKEN"       `# Optionally require a token for access` \
        --NotebookApp.password=""                  `# Do not require password to access the course` \
        --MultiKernelManager.default_kernel_name="reopt"
