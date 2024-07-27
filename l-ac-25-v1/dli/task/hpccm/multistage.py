# Multistage recipe
# Note that the base images are different.  The development stage uses
# the "devel" CUDA base image, while the deployment stage uses the smaller
# "base" image.
Stage0 += baseimage(image='nvidia/cuda:9.0-devel-centos7')
Stage0 += gnu()

Stage1 += baseimage(image='nvidia/cuda:9.0-base-centos7')
Stage1 += Stage0.runtime()
