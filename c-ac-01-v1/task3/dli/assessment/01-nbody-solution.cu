#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

__global__
void bodyForce(Body *p, float dt, int n) {

  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}


__global__
void integratePositions(Body *p, float dt, int n) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
}


int main(const int argc, const char** argv) {

  int nBodies = 2<<atoi(argv[1]);
  const char * initialized_values = argv[2];
  const char * solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  cudaMallocManaged(&buf, bytes);

  Body *p = (Body*)buf;

  read_values_from_file(initialized_values, buf, bytes);

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaMemPrefetchAsync(p, bytes, deviceId);

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

  /*
   * You will likely wish to refactor the work being done in `bodyForce`,
   * and potentially the work to integrate the positions.
   */

    size_t threads_per_block = 256;
    size_t number_of_blocks = (nBodies + threads_per_block - 1) / threads_per_block;

    bodyForce<<<number_of_blocks, threads_per_block>>>(p, dt, nBodies); // compute interbody forces
    integratePositions<<<number_of_blocks, threads_per_block>>>(p, dt, nBodies); // compute interbody forces

    cudaDeviceSynchronize();
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  printf("%0.3f Billion Interactions / second", billionsOfOpsPerSecond);

  cudaFree(buf);
}
