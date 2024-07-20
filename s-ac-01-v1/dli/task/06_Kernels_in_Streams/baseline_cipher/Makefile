CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3
CXXFLAGS=-march=native -fopenmp
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

all: baseline

baseline: baseline.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" baseline.cu -o baseline

profile: baseline
	$(NSYS) $(NSYSFLAGS) -o baseline-report ./baseline

baseline_solution: baseline_solution.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" baseline_solution.cu -o baseline_solution

profile_solution: baseline_solution
	$(NSYS) $(NSYSFLAGS) -o baseline-solution-report ./baseline_solution

clean:
	rm -f baseline baseline_solution *.qdrep *.sqlite
