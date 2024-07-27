Stage0 += baseimage(image='nvidia/cuda:9.2-devel-ubuntu16.04')
Stage0 += openmpi(configure_opts=['--disable-getpwuid',
                                  '--enable-orterun-prefix-by-default',
                                  '--disable-fortran'],
                  infiniband=False, prefix='/opt/openmpi', version='2.1.2')
