# CentOS base image
Stage0 += baseimage(image='centos:7.6.1810')

# GNU compilers
Stage0 += gnu(fortran=False)

# Mellanox OFED
Stage0 += mlnx_ofed(version='4.5-1.0.1.0')

# OpenMPI
Stage0 += openmpi(cuda=False, version='3.1.2')

# MPI Bandwidth
Stage0 += copy(src='sources/mpi_bandwidth.c', dest='/var/tmp/mpi_bandwidth.c')
Stage0 += shell(commands=[
    'mpicc -o /usr/local/bin/mpi_bandwidth /var/tmp/mpi_bandwidth.c'])

###
# Runtime stage
###

# CentOS base image
Stage1 += baseimage(image='centos:7.6.1810')

# Runtime versions of all components from the previous stage
Stage1 += Stage0.runtime()

# MPI Bandwidth
Stage1 += copy(_from='0', src='/usr/local/bin/mpi_bandwidth',
               dest='/usr/local/bin/mpi_bandwidth')
