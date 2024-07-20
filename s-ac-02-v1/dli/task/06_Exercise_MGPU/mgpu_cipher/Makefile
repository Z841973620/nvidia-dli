CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

all: mgpu

mgpu: mgpu.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mgpu.cu -o mgpu

mgpu_solution: mgpu_solution.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" mgpu_solution.cu -o mgpu_solution

profile: mgpu
	$(NSYS) $(NSYSFLAGS) -o mgpu-report ./mgpu

profile_solution: mgpu_solution
	$(NSYS) $(NSYSFLAGS) -o mgpu-solution-report ./mgpu_solution

clean:
	rm -f mgpu mgpu_solution *.qdrep *.sqlite
