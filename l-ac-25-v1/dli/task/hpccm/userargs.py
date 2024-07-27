"""This example demonstrates user arguments.

The CUDA and OpenMPI versions can be specified on the command line.  If
they are not, then reasonable defaults are used.

Usage:
$ hpccm --recipe userargs.py --userarg cuda=9.0 ompi=2.1.2
"""

from distutils.version import StrictVersion

cuda_version = USERARG.get('cuda', '9.1')
if StrictVersion(cuda_version) < StrictVersion('9.0'):
  raise RuntimeError('invalid CUDA version: {}'.format(cuda_version))
Stage0 += baseimage(image='nvidia/cuda:{}-devel-ubuntu16.04'.format(cuda_version))

ompi_version = USERARG.get('ompi', '3.1.2')
if not StrictVersion(ompi_version):
  raise RuntimeError('invalid OpenMPI version: {}'.format(ompi_version))
Stage0 += openmpi(infiniband=False, version=ompi_version)
