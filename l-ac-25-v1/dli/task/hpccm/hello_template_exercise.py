Stage0 += baseimage(image='centos:7')
Stage0 += gnu(fortran=False)
Stage0 += comment('Insert your content below')
Stage0 += comment('First, copy hello.c from the host into the container')
Stage0 += comment('Then compile the source code: gcc -o /usr/local/bin/hello /var/tmp/hello.c')