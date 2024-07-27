#!/usr/bin/env python3

from __future__ import print_function

import argparse
import hpccm
from hpccm.building_blocks import gnu, mlnx_ofed, openmpi
from hpccm.primitives import baseimage

parser = argparse.ArgumentParser(description='HPCCM Tutorial')
parser.add_argument('--format', type=str, default='docker',
                    choices=['docker', 'singularity'],
                    help='Container specification format (default: docker)')
args = parser.parse_args()

Stage0 = hpccm.Stage()

### Start "Recipe"
Stage0 += baseimage(image='nvidia/cuda:9.0-devel-centos7')
Stage0 += mlnx_ofed(version='3.4-1.0.0.0')
compiler = gnu()
Stage0 += compiler
Stage0 += openmpi(cuda=True, infiniband=True, toolchain=compiler.toolchain,
                 version='1.10.7')
### End "Recipe"

hpccm.config.set_container_format(args.format)

print(Stage0)
