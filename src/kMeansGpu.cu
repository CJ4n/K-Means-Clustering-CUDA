#include "kMeansGpu.h"
#include "findClosestCentriods.h"
#include "cuda.h"
#include "cudaCheckError.h"

// __global__ void ReducePoints(DataPoints *points, DataPoints *centroids)
// {
// 	int tid = threadIdx.x;
// 	int gid = blockIdx.x * blockDim.x + threadIdx.x;
// }

// template <int NUM_FEATURES=2,int NUM_DATA_POINTS=200>
__global__ void reduce4(DataPoints *points, DataPoints *centroids, DataPoints **out)
{
	extern __shared__ float shm[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int feature_stirde = centroids->num_data_points;

	for (int f = 0; f < centroids->num_features; f++)
	{

		if (gid + blockDim.x >= points->num_data_points)
		{
			break;
		}

		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			shm[(f * feature_stirde) + c + tid * centroids->num_features * centroids->num_data_points] = 0;
		}

		int c = points->cluster_id_of_point[gid + blockDim.x];
		shm[(f * feature_stirde) + c + tid * centroids->num_features * centroids->num_data_points] = points->features_array[f][gid + blockDim.x];

		c = points->cluster_id_of_point[gid + blockDim.x];
		shm[(f * feature_stirde) + c + tid * centroids->num_features * centroids->num_data_points] = points->features_array[f][gid + blockDim.x];
		// idx where to store particualr feature coord
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
		{
			for (int f = 0; f < centroids->num_features; ++f)
			{
				for (int c = 0; c < centroids->num_data_points; ++c)
				{
					shm[(f * feature_stirde) + c + tid * centroids->num_features * centroids->num_data_points] += shm[(f * feature_stirde) + c + (tid + stride) * centroids->num_features * centroids->num_data_points];
				}
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int f = 0; f < centroids->num_features; f++)
		{
			for (int c = 0; c < centroids->num_data_points; c++)
			{
				out[blockIdx.x]->features_array[f][c] = shm[(f * feature_stirde + c)];
			}
		}
		// out[blockIdx.x]= shm[3];
	}
}
__global__ void reduce4123(float *in, float *out, int N)
{
	extern __shared__ float shm[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	shm[tid] = in[gid] + in[gid + blockDim.x];
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
		{
			shm[tid] += shm[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		out[blockIdx.x] = shm[0];
	}
}
#include <iostream>

void KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{

	int N = points->num_data_points;
	int num_threads = 1024;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	size_t shmem_size = (num_threads) * sizeof(float) * centroids->num_features * centroids->num_data_points;
		FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);
		cudaDeviceSynchronize();

	DataPoints **out;
	int ile = std::ceil(num_blocks / 2);
	cudaMallocManaged(&out, sizeof(DataPoints *) * ile);
	for (int i = 0; i < ile; i++)
	{
		out[i] = AllocateDataPoints(centroids->num_features, centroids->num_data_points);
	}

	reduce4 /*<centroids->num_features,centroids->num_data_points>*/<<<num_blocks / 2, num_threads, shmem_size>>>(points, centroids, out);
	cudaDeviceSynchronize();
	cudaCheckError();

	for (int i = ile - 1; i < ile; i++)
	{
		std::cout << "block id: " << i << std::endl;
		for (int f = 0; f < out[i]->num_features; f++)
		{
			for (int c = 0; c < out[i]->num_data_points; c++)
			{
				std::cout << out[i]->features_array[f][c] << ", ";
			}
		}
		std::cout << "\nblock id: " << i << std::endl;
	}

	for (int i = 0; i < ile; i++)
	{
		DeallocateDataPoints(out[i]);
	}
	cudaFree(out);
	std::cout << std::endl;

	cudaCheckError();
}