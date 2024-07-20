import numpy as np
from numba import jit, cuda

x = np.random.normal(size=10000, loc=0, scale=1).astype(np.float32)
xmin = np.float32(-4.0)
xmax = np.float32(4.0)
histogram_out = np.zeros(shape=10, dtype=np.int32)

d_x = cuda.to_device(x)
d_histogram_out = cuda.device_array_like(histogram_out)

@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        bin_number = np.int32((x[i] - xmin)/bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            cuda.atomic.add(histogram_out, bin_number, 1)

blocks = 128
threads_per_block = 64

cuda_histogram[blocks, threads_per_block](d_x, xmin, xmax, d_histogram_out)
cuda.synchronize()

solution_x = np.random.normal(size=10000, loc=0, scale=1).astype(np.float32)
solution_xmin = np.float32(-4.0)
solution_xmax = np.float32(4.0)
solution_histogram_out = np.zeros(shape=10, dtype=np.int32)

solution_d_x = cuda.to_device(x)
solution_d_histogram_out = cuda.device_array_like(solution_histogram_out)

@cuda.jit
def solution_cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        bin_number = np.int32((x[i] - xmin)/bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            cuda.atomic.add(histogram_out, bin_number, 1)


solution_blocks = 128
solution_threads_per_block = 64

solution_cuda_histogram[solution_blocks, solution_threads_per_block](solution_d_x, solution_xmin, solution_xmax, solution_d_histogram_out)
cuda.synchronize()

if type(d_histogram_out) == cuda.cudadrv.devicearray.DeviceNDArray:
    student_result = d_histogram_out.copy_to_host()
else:
    student_result = d_histogram_out

solution_result = solution_d_histogram_out.copy_to_host()
is_correct = np.array_equal(student_result, solution_result)
print('XXX', is_correct) # We will use XXX later to identify this as assessment output
