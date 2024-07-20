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
