#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {

    Timer timer;
    Timer overall;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;

    const uint64_t num_streams = 32;
    const uint64_t chunk_size = sdiv(num_entries, num_streams);

    timer.start();
    uint64_t * data_cpu, * data_gpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);
    timer.stop("allocate memory");
    check_last_error();

    timer.start();
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
    timer.stop("encrypt data on CPU");

    timer.start();
    cudaStream_t streams[num_streams];
    for (uint64_t stream = 0; stream < num_streams; stream++)
        cudaStreamCreate(&streams[stream]);
    timer.stop("create streams");
    check_last_error();

    overall.start();
    timer.start();
    for (uint64_t stream = 0; stream < num_streams; stream++) {
        
        const uint64_t lower = chunk_size*stream;
        const uint64_t upper = min(lower+chunk_size, num_entries);
        const uint64_t width = upper-lower;

        cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
               sizeof(uint64_t)*width, cudaMemcpyHostToDevice, 
               streams[stream]);
    
        decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>
            (data_gpu+lower, width, num_iters);

        cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
               sizeof(uint64_t)*width, cudaMemcpyDeviceToHost, 
               streams[stream]);
    }    
    timer.stop("asynchronous H2D->kernel->D2H");
    overall.stop("total time on GPU");
    check_last_error();
    
    timer.start();
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;
    timer.stop("checking result on CPU");

    timer.start();
    for (uint64_t stream = 0; stream < num_streams; stream++)
        cudaStreamDestroy(streams[stream]);    
    timer.stop("destroy streams");
    check_last_error();

    timer.start();
    cudaFreeHost(data_cpu);
    cudaFree    (data_gpu);
    timer.stop("free memory");
    check_last_error();
}
