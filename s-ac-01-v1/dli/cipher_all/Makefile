CUDACXX=nvcc
CUDACXXFLAGS=-arch=sm_70 -O3
CXXFLAGS=-march=native -fopenmp

all: baseline streams multigpu multigpu_fused multigpu_streams

baseline: baseline.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" baseline.cu -o baseline

streams: streams.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" streams.cu -o streams

multigpu: multigpu.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" multigpu.cu -o multigpu

multigpu_fused: multigpu_fused.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" multigpu_fused.cu -o multigpu_fused

multigpu_streams: multigpu_streams.cu
	$(CUDACXX) $(CUDACXXFLAGS) -Xcompiler="$(CXXFLAGS)" multigpu_streams.cu -o multigpu_streams

clean:
	rm -f baseline
	rm -f streams
	rm -f multigpu
	rm -f multigpu_fused
	rm -f multigpu_streams
