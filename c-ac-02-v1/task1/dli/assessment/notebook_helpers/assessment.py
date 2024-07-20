import numpy as np
from numba import cuda, vectorize
from math import exp as solution_exp
from time import perf_counter

def assess(create_hidden_layer, arguments):

    print("Setting n to 100 million.")
    n = 100000000

    greyscales = np.floor(np.random.uniform(0, 255, n).astype(np.float32))
    weights = np.random.normal(.5, .1, n).astype(np.float32)

    solution_greyscales = greyscales[:]
    solution_weights = weights[:]

    student_greyscales = greyscales[:]
    student_weights = weights[:]
    
    arguments["n"] = n
    arguments["greyscales"] = student_greyscales
    arguments["weights"] = student_weights
    
    student_start = perf_counter()
    a = create_hidden_layer(**arguments)
    student_end = perf_counter()
    student_total = student_end - student_start
    
    is_host_array = type(a) is np.ndarray
    print("\nYour function returns a host np.ndarray:", is_host_array)
    if not is_host_array:
        print("Please refactor your code to return a host array and try again.")
        return
    
    target_time = 1
    fast_enough = student_total < target_time
    print("\nYour function took {:0.2f}s to run.".format(student_total))
    print("Your function runs fast enough (less than {} second): {}".format(target_time, fast_enough))
    if not fast_enough:
        print("Please refactor your code to run faster and try again.")
        return
    
    @vectorize(['float32(float32)'], target='cuda')
    def solution_normalize(grayscales):
        return grayscales / 255

    @vectorize(['float32(float32, float32)'], target='cuda')
    def solution_weigh(values, weights):
        return values * weights

    @vectorize(['float32(float32)'], target='cuda')
    def solution_activate(values):
        return ( solution_exp(values) - solution_exp(-values) ) / ( solution_exp(values) + solution_exp(-values) )

    def solution_create_hidden_layer(greyscales, weights, exp, normalize, weigh, activate):
        normalized = cuda.device_array_like(greyscales)
        weighted = cuda.device_array_like(greyscales)
        activated = cuda.device_array_like(greyscales)

        normalize(greyscales, out=normalized)
        weigh(normalized, weights, out=weighted)
        activate(weighted, out=activated)
        cuda.synchronize()

        return activated.copy_to_host()

    solution_arguments = {"greyscales": solution_greyscales,
                          "weights": solution_weights,
                          "exp": solution_exp,
                          "normalize": solution_normalize,
                          "weigh": solution_weigh,
                          "activate": solution_activate}
    
    solution_a = solution_create_hidden_layer(**solution_arguments)

    is_correct = np.allclose(solution_a, a, atol=.01)
    print("\nYour function returns the correct results:", is_correct)
    if not is_correct:
        print("Your function is not returning the correct result. Please fix and try again.")
        return
    
    print("Congratulations, you passed!")
