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
__global__ void reduce4(DataPoints *points, int k /*number of centroids*/, DataPoints *out)
{
	extern __shared__ float shm[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int feature_stirde = k;

	// shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]

	for (int f = 0; f < points->num_features; ++f)
	{

		if (gid + blockDim.x >= points->num_data_points)
		{
			return;
		}

		for (int c = 0; c < k; ++c)
		{
			shm[((f * feature_stirde) + c) + tid * points->num_features * k] = 0;
		}

		int c = points->cluster_id_of_point[gid];
		shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid];

		c = points->cluster_id_of_point[gid + blockDim.x];
		shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid + blockDim.x];
		// idx where to store particualr feature coord
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
		{
			for (int f = 0; f < points->num_features; ++f)
			{
				for (int c = 0; c < k; ++c)
				{
					shm[((f * feature_stirde) + c) + tid * points->num_features * k] += shm[(f * feature_stirde) + c + (tid + stride) * points->num_features * k];
				}
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int f = 0; f < points->num_features; ++f)
		{
			for (int c = 0; c < k; ++c)
			{
				out->features_array[f][c + blockIdx.x * k] = shm[(f * feature_stirde + c)];
			}
		}
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
	int num_threads = 1024/4;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	size_t shmem_size = (num_threads) * sizeof(float) * centroids->num_features * centroids->num_data_points;
	FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);
	cudaDeviceSynchronize();

	DataPoints *out;
	num_blocks = std::ceil(num_blocks / 2);

	out = AllocateDataPoints(centroids->num_features, centroids->num_data_points * num_blocks);
	cudaCheckError();

	for(int i=0;i<centroids->num_data_points*num_blocks;i++){
		out->cluster_id_of_point[i]=i%centroids->num_data_points;
	}

	std::cout << "summed without kernel:  ";

	for (int f = 0; f < points->num_features; f++)
	{
		for (int c = 0; c < centroids->num_data_points; c++)
		{
			int sum = 0;
			for (int i = 0; i < points->num_data_points; i++)
			{
				if (c == points->cluster_id_of_point[i])
					sum += points->features_array[f][i];
			}
			std::cout << sum << ", ";
		}
	}
	std::cout << std::endl;
	cudaCheckError();

	reduce4 /*<centroids->num_features,centroids->num_data_points>*/<<<num_blocks, num_threads, shmem_size>>>(points, centroids->num_data_points, out);
	cudaDeviceSynchronize();
	cudaCheckError();

	std::cout << "summed with one kernel: ";

	for (int f = 0; f < out->num_features; f++)
	{
		for (int c = 0; c < centroids->num_data_points; c++)
		{
			int sum = 0;
			for (int i = 0; i < num_blocks; i++)
				sum += out->features_array[f][c + i * centroids->num_data_points];
			std::cout << sum << ", ";
		}
	}
	// std::cout << std::endl;
	// for (int i = 0; i < 1; i++)
	// {
	// 	std::cout << "blockId: " << i << std::endl;
	// 	for (int f = 0; f < out->num_features; f++)
	// 	{
	// 		for (int c = 0; c < centroids->num_data_points; c++)
	// 		{
	// 			std::cout << out->features_array[f][c + i * centroids->num_data_points] << ", ";
	// 		}
	// 	}

	// 	std::cout << std::endl;
	// }
	std::cout << std::endl;
	
	num_threads = num_blocks * centroids->num_data_points/2;
	shmem_size = num_threads* sizeof(float) * centroids->num_features * centroids->num_data_points;
	// jest jakiś problem z sumowanien dla niektórych clustrów?>?
	// wyklada że problem gdy liczba kalstrow to nie wielokrotnosci 2
	reduce4 /*<centroids->num_features,centroids->num_data_points>*/<<<1, num_threads, shmem_size>>>(out, centroids->num_data_points, out);
	cudaDeviceSynchronize();
	cudaCheckError();
	// for (int i = 0; i < 1; i++)
	// {
	// 	std::cout << "blockId: " << i << std::endl;
	// 	for (int f = 0; f < out->num_features; f++)
	// 	{
	// 		for (int c = 0; c < centroids->num_data_points; c++)
	// 		{
	// 			std::cout << out->features_array[f][c + i * centroids->num_data_points] << ", ";
	// 		}
	// 	}

	// 	std::cout << std::endl;
	// }
	std::cout<<"sumed all with kernel:  ";
	for (int f = 0; f < out->num_features; f++)
	{
		for (int c = 0; c < centroids->num_data_points; c++)
		{
			int sum = 0;
			for (int i = 0; i < 1; i++)
				sum += out->features_array[f][c + i * centroids->num_data_points];
			std::cout << sum << ", ";
		}
	}

	std::cout << std::endl;

	DeallocateDataPoints(out);
	cudaCheckError();
}