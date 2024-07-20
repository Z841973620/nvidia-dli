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

    const uint64_t num_gpus = 4;
    const uint64_t num_streams = 32;
    const uint64_t chunk_size = sdiv(sdiv(num_entries, num_gpus), num_streams);

    cudaStream_t streams[num_gpus][num_streams];

    timer.start();
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++)
            cudaStreamCreate(&streams[gpu][stream]);
    }
    timer.stop("create streams");
    check_last_error();


    timer.start();
    uint64_t * data_cpu, * data_gpu[num_gpus];
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {

        cudaSetDevice(gpu);

        const uint64_t lower = chunk_size*num_streams*gpu;
        const uint64_t upper = min(lower+chunk_size*num_streams, num_entries);
        const uint64_t width = upper-lower;

        cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width);
    }
    timer.stop("allocate memory");
    check_last_error();

    timer.start();
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
    timer.stop("encrypt data on CPU");

    overall.start();
    timer.start();
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++) {


            const uint64_t offset = chunk_size*stream;
            const uint64_t lower = chunk_size*num_streams*gpu+offset;
            const uint64_t upper = min(lower+chunk_size, num_entries);
            const uint64_t width = upper-lower;

            cudaMemcpyAsync(data_gpu[gpu]+offset, data_cpu+lower, 
                            sizeof(uint64_t)*width, cudaMemcpyHostToDevice,
                            streams[gpu][stream]);
    
            decrypt_gpu<<<80*32, 64, 0, streams[gpu][stream]>>>
                (data_gpu[gpu]+offset, width, num_iters);
    
            cudaMemcpyAsync(data_cpu+lower, data_gpu[gpu]+offset, 
                            sizeof(uint64_t)*width, cudaMemcpyDeviceToHost,
                            streams[gpu][stream]);
        }
    }

    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++) {
            cudaStreamSynchronize(streams[gpu][stream]);
        }
    }
    timer.stop("asynchronous H2D -> kernel -> D2H multiGPU");
    overall.stop("total time on GPU");
    check_last_error();

    timer.start();
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;
    timer.stop("checking result on CPU");

    timer.start();
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++) {
            cudaStreamDestroy(streams[gpu][stream]);
        }
    }
    timer.stop("destroy streams");
    check_last_error();

    timer.start();
    cudaFreeHost(data_cpu);
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(data_gpu[gpu]);
    }    
    timer.stop("free memory");
    check_last_error();
}
