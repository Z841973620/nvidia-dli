import subprocess
import time
import numpy as np

all_inputs_and_solutions = {
  4096: [
    ('/dli/assessment/test_files/initialized_4096_A', '/dli/assessment/test_files/solution_4096_A')
  ],
  65536: [
    ('/dli/assessment/test_files/initialized_65536_A', '/dli/assessment/test_files/solution_65536_A')
  ]
}

expected_runtimes = {
  'GPU': {
    '11': .85,
    '15': 1.0,
  }
}

def compare_files(a, b):
  # Floating point calculations between the CPU and GPU vary, so in order to 
  # be able to assess both CPU-only and GPU code we compare the floating point
  # arrays within a tolerance of less than 1% of values differ by 1 or more.
  
  file_a = np.fromfile(a, dtype=np.float32)
  file_b = np.fromfile(b, dtype=np.float32)
  
  c = np.abs(file_a - file_b)
  d = c[np.where(c > 1)]
  
  return (len(d)/len(file_a)) < .01


def passes_with_n_of(n):
  student_output = '/dli/assessment/test_files/student_output'
  nbodies = 2<<int(n)
  inputs_and_solutions = all_inputs_and_solutions[nbodies]

  print('使用{}个物体运行n-体模拟器'.format(nbodies))
  print('----------------------------------------\n')

  expected_runtime = expected_runtimes['GPU'][n]
  print('此应用程序应该运行得快于{}秒。'.format(expected_runtime))

  for input, solution in inputs_and_solutions:
    start = time.perf_counter()

    try:
      p = subprocess.run(['/dli/task/nbody', n, input, student_output], capture_output=True, text=True, timeout=5)
      ops_per_second = p.stdout
    except:
      print('您的应用程序已运行了5秒钟以上的时间，运行速度不够快。')
      return False

    end = time.perf_counter()
  
    actual_runtime = end-start

    print('您的应用程序运行了: {:.4f}秒'.format(actual_runtime))
    if actual_runtime > expected_runtime:
      print('您的应用程序仍不够快。')
      return False

    print('您的应用程序运行速度是 ', ops_per_second)

    correct = compare_files(solution, student_output)
    if correct:
      print('您的结果是正确的。\n')
    else:
      print('您的结果是不正确的。\n')
      return False

  return True


def run_assessment():
  if passes_with_n_of('11') is False:
    return
  if passes_with_n_of('15') is False:
    return
  print('祝贺您！您通过了评估！')
