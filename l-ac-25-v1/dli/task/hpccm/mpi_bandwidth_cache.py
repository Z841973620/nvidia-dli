# Used to create a Docker layer cache in order to speed up
# the associated container build during the lab

# CentOS base image
Stage0 += baseimage(image='centos:7.6.1810')

# GNU compilers
Stage0 += gnu(fortran=False)

# Mellanox OFED
Stage0 += mlnx_ofed(version='4.5-1.0.1.0')

# OpenMPI
Stage0 += openmpi(cuda=False, version='3.1.2')
