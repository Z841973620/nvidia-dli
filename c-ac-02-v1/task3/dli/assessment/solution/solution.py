@cuda.jit
def tile_transpose(a, transposed):
    # `tile_transpose` assumes it is launched with a 32x32 block dimension,
    # and that `a` is a multiple of these dimensions.
    
    # 1) Create 32x32 shared memory array.
    tile = cuda.shared.array((32, 33), numba_types.int32)

    # Compute offsets into global input array.
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # 2) Make coalesced read from global memory into shared memory array.
    # Note the use of local thread indices for the shared memory write,
    # and global offsets for global memory read.
    tile[cuda.threadIdx.y, cuda.threadIdx.x] = a[col, row]

    # 3) Wait for all threads in the block to finish updating shared memory.
    cuda.syncthreads()
    
    # 4) Calculate transposed location for the shared memory array tile
    # to be written back to global memory.
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.x
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.y

    # 5) Write back to global memory,
    # transposing each element within the shared memory array.
    transposed[col, row] = tile[cuda.threadIdx.x, cuda.threadIdx.y]
