# Development stage containing the development environment and source code.
Stage0 += baseimage(image='centos:7')
Stage0 += gnu(fortran=False)

# Use HPCCM primitives to copy the source into the container image
# and then compile the application.
Stage0 += copy(src='sources/hello.c', dest='/var/tmp/hello.c')
Stage0 += shell(commands=['gcc -o /usr/local/bin/hello /var/tmp/hello.c'])

#################

# Runtime stage containing only the required runtime and binary
Stage1 += baseimage(image='centos:7')

# Include GNU compiler runtime components
Stage1 += Stage0.runtime()

# Copy the binary from the previous stage, "_from='0'".  '0' refers
# to the first stage above.
Stage1 += copy(_from='0', src='/usr/local/bin/hello', dest='/usr/local/bin/hello')
