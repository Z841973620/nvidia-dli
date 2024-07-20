import numpy as np
from numba import jit, cuda

x = np.random.normal(size=10000, loc=0, scale=1).astype(np.float32)
xmin = np.float32(-4.0)
xmax = np.float32(4.0)
histogram_out = np.zeros(shape=10, dtype=np.int32)

d_x = cuda.to_device(x)
d_histogram_out = cuda.device_array_like(histogram_out)
