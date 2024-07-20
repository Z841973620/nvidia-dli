if type(d_histogram_out) == cuda.cudadrv.devicearray.DeviceNDArray:
    student_result = d_histogram_out.copy_to_host()
else:
    student_result = d_histogram_out

solution_result = solution_d_histogram_out.copy_to_host()
is_correct = np.array_equal(student_result, solution_result)
print('XXX', is_correct) # We will use XXX later to identify this as assessment output
