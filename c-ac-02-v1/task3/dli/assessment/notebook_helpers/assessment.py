import numpy as np
from numba import cuda, types as numba_types
from time import perf_counter

@cuda.jit
def tile_transpose(a, transposed):
    tile = cuda.shared.array((32, 32), numba_types.int32)

    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    tile[cuda.threadIdx.y, cuda.threadIdx.x] = a[y, x]

    cuda.syncthreads()
    
    t_x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.x
    t_y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.y

    transposed[t_y, t_x] = tile[cuda.threadIdx.x, cuda.threadIdx.y]

def target_time(target_transpose):

    # Define data and grid/block dimensions in the same wasy as the notebook.
    n = 4096*4096 # 16M

    threads_per_block = (32, 32)
    blocks = (128, 128)

    a = np.arange(n).reshape((4096,4096)).astype(np.float32)
    transposed = np.zeros_like(a).astype(np.float32)

    d_a = cuda.to_device(a)
    d_transposed = cuda.to_device(transposed)
    

    # Run student code 10k times and get average, to mimic %timeit performance in notebook.
    target_total = 0
    num_runs = 10000
    for _ in range(num_runs):

        target_start = perf_counter()

        target_transpose[blocks, threads_per_block](d_a, d_transposed); cuda.synchronize()

        target_end = perf_counter()
        target_total += target_end - target_start # This value in in seconds.

    target_total /= num_runs
    target_total_microseconds = target_total * 1e6 # target_total is in seconds.
    
    return target_total_microseconds

def assess(student_transpose):

    # Define data and grid/block dimensions in the same wasy as the notebook.
    n = 4096*4096 # 16M

    threads_per_block = (32, 32)
    blocks = (128, 128)

    a = np.arange(n).reshape((4096,4096)).astype(np.float32)
    transposed = np.zeros_like(a).astype(np.float32)

    d_a = cuda.to_device(a)
    d_transposed = cuda.to_device(transposed)
    

    # Run student code 10k times and get average, to mimic %timeit performance in notebook.
    student_total = 0
    num_runs = 10000
    for _ in range(num_runs):

        student_start = perf_counter()

        student_transpose[blocks, threads_per_block](d_a, d_transposed); cuda.synchronize()

        student_end = perf_counter()
        student_total += student_end - student_start # This value in in seconds.

    student_total /= num_runs
    

    # Compare target and student performance.
    target_time_microseconds = target_time(tile_transpose) - 10
    student_total_microseconds = student_total * 1e6 # student_total is in seconds.
    fast_enough = student_total_microseconds <= target_time_microseconds # Less than or equal to 210 µs, as indicated in the notebook


    # Report to students about performance.
    print("\nYour function took {:0.2f} µs to run.\n".format(student_total_microseconds))
    print("Your function runs fast enough (less than {} µs): {}\n".format(target_time_microseconds, fast_enough))

    if not fast_enough:
        print("Please refactor your code to run faster and try again.\n")
        return


    # Check student code for correctness.
    result = d_transposed.copy_to_host()
    expected = a.T # Simple numpy transpose gives expected result for student transpose kernel.
    is_correct = np.array_equal(expected, result)


    # Report to students about correctness.
    print("Your function returns the correct results: {}\n".format(is_correct))
    if not is_correct:
        print("Your function is not returning the correct result. Please fix and try again.")
        return
    

    # If successful, report so back to students.
    print("Congratulations, you passed!")
