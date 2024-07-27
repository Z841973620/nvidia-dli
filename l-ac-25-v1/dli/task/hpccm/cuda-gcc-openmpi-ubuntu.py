Stage0 += baseimage(image='nvidia/cuda:9.0-devel-ubuntu16.04')
Stage0 += mlnx_ofed(version='3.4-1.0.0.0')
compiler = gnu()
Stage0 += compiler
Stage0 += openmpi(cuda=True, infiniband=True, toolchain=compiler.toolchain,
                  version='1.10.7')
