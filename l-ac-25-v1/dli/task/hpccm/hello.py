Stage0 += baseimage(image='centos:7')
Stage0 += gnu(fortran=False)

# Use HPCCM primitives to copy the source into the container image
# and then compile the application.
Stage0 += copy(src='sources/hello.c', dest='/var/tmp/hello.c')
Stage0 += shell(commands=['gcc -o /usr/local/bin/hello /var/tmp/hello.c'])
