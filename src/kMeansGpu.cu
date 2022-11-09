#include "kMeansGpu.h"
#include "findClosestCentriods.h"
#include "cuda.h"
#include "cudaCheckError.h"

void KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{
    int N = points->num_data_points;
    int num_threads = 1024;
    int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
    // size_t shmem_size = num_threads * sizeof(float);

    FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
     
}