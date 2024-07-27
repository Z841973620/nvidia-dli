# The development stage contains the development environment and source code.
# The runtime stage contains only the required runtime and binary.

# Base images - use the same for both stages
Stage0 += baseimage(image='centos:7')
Stage1 += baseimage(image='centos:7')

# GNU compiler
g = gnu(fortran=False)
Stage0 += g
Stage1 += g.runtime()

# Use HPCCM primitives to copy the source into the container image
# and then compile the application.  Copy the resulting binary to the
# runtime stage.
Stage0 += copy(src='sources/hello.c', dest='/var/tmp/hello.c')
Stage0 += shell(commands=['gcc -o /usr/local/bin/hello /var/tmp/hello.c'])
Stage1 += copy(_from='0', src='/usr/local/bin/hello', dest='/usr/local/bin/hello')
