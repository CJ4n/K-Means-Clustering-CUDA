#include "kMeansGpu.h"
#include "findClosestCentriods.h"
#include "cuda.h"
#include "cudaCheckError.h"

// template <int NUM_FEATURES=2,int NUM_DATA_POINTS=200>
__global__ void reduce4(const DataPoints *points, int k /*number of centroids*/, DataPoints *out, int num_threads, int num_blocks)
{
	extern __shared__ float shm[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int feature_stirde = k;

	// shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]
	for (int f = 0; f < points->num_features; ++f)
	{

		for (int c = 0; c < k; ++c)
		{
			shm[((f * feature_stirde) + c) + tid * points->num_features * k] = 0;
		}
		if (gid + blockDim.x >= points->num_data_points)
		{
			break;
		}

		int c = points->cluster_id_of_point[gid];

		shm[((f * feature_stirde) + c) + tid * points->num_features * k] = points->features_array[f][gid];

		c = points->cluster_id_of_point[gid + blockDim.x];
		shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid + blockDim.x];
		// idx where to store particualr feature coord
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		// problem jest gdy num_threads nie jest wiekokrotnoscia 2, wtedy jak mamy blockdim.x/2 itd, dostanimy cos co sie nie podzieli przez 2
		if (tid < stride)
		{
			for (int f = 0; f < points->num_features; ++f)
			{
				for (int c = 0; c < k; ++c)
				{
					// czy to jest optumalny odczyt??
					shm[((f * feature_stirde) + c) + tid * points->num_features * k] += shm[((f * feature_stirde) + c) + (tid + stride) * points->num_features * k];
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
				out->features_array[f][c + blockIdx.x * k] = shm[((f * feature_stirde) + c)];
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
	int num_threads = 1024 / 4;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	size_t shmem_size = num_threads * sizeof(float) * points->num_features * centroids->num_data_points;
	FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);
	cudaDeviceSynchronize();

	num_blocks = std::ceil(num_blocks / 2);
	auto lambda = [](int n)
	{
		unsigned count = 0;
		if (n && !(n & (n - 1)))
			return n;

		while (n != 0)
		{
			n >>= 1;
			count += 1;
		}

		return 1 << count;
	};
	int tmp = num_blocks * centroids->num_data_points;
	tmp = lambda(tmp);

	DataPoints *out = AllocateDataPoints(centroids->num_features, tmp);
	cudaCheckError();

	for (int i = 0; i < tmp; i++)
	{
		out->cluster_id_of_point[i] = i % centroids->num_data_points;
	}

	std::cout << "summed without kernel:  ";

	for (int c = 0; c < centroids->num_data_points; c++)
	{
		for (int f = 0; f < points->num_features; f++)
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

	reduce4 /*<centroids->num_features,centroids->num_data_points>*/<<<num_blocks, num_threads, shmem_size>>>(points, centroids->num_data_points, out, num_threads, num_blocks);
	cudaDeviceSynchronize();
	cudaCheckError();

	std::cout << "summed with one kernel: ";

	for (int c = 0; c < centroids->num_data_points; c++)
	{
		for (int f = 0; f < out->num_features; f++)
		{
			int sum = 0;
			for (int i = 0; i < tmp; i++)
				if (out->cluster_id_of_point[i] == c)
					sum += out->features_array[f][i];
			std::cout << sum << ", ";
		}
	}
	std::cout << std::endl;

	num_threads = tmp / 2;

	shmem_size = num_threads * sizeof(float) * centroids->num_features * centroids->num_data_points;
	// jest jakiś problem z sumowanien dla niektórych clustrów?>?
	// wyklada że problem gdy liczba kalstrow to nie wielokrotnosci 2
	auto ret = AllocateDataPoints(centroids->num_features, centroids->num_data_points);
	reduce4 /*<centroids->num_features,centroids->num_data_points>*/<<<1, num_threads, shmem_size>>>(out, centroids->num_data_points, ret, num_threads, 1);
	cudaDeviceSynchronize();
	cudaCheckError();

	// std::cout << "sumed all with kernel:  ";
	// for (int c = 0; c < centroids->num_data_points; c++)
	// {
	// 	for (int f = 0; f < out->num_features; f++)
	// 	{
	// 		int sum = 0;
	// 		for (int i = 0; i < 1; i++)
	// 			sum += out->features_array[f][c + i * centroids->num_data_points];
	// 		std::cout << sum << ", ";
	// 	}
	// }
	// for (int c = 0; c < centroids->num_data_points; c++)
	// {
	// 	std::cout << "cluster: " << c << "|";
	// 	for (int f = 0; f < points->num_features; f++)
	// 	{
	// 		int sum = 0;
	// 		for (int i = 0; i < points->num_data_points; i++)
	// 		{
	// 			if (c == points->cluster_id_of_point[i])
	// 				sum += points->features_array[f][i];
	// 		}
	// 		std::cout << sum << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << "summed with one kernel: ";

	// for (int f = 0; f < out->num_features; f++)
	// {
	// 	for (int c = 0; c < centroids->num_data_points; c++)
	// 	{
	// 		int sum = 0;
	// 		for (int i = 0; i < num_blocks; i++)
	// 			sum += out->features_array[f][c + i * centroids->num_data_points];
	// 		std::cout << sum << ", ";
	// 	}
	// }
	// std::cout << std::endl;

	for (int c = 0; c < centroids->num_data_points; c++)
	{
		// std::cout << "cluster: " << c << "|";

		std::cout << "cluster: " << c << "|\n";
		std::cout << "summed without kernel:  ";

		for (int f = 0; f < points->num_features; f++)
		{
			int sum = 0;
			for (int i = 0; i < points->num_data_points; i++)
			{
				// if (c == points->cluster_id_of_point[i])
					sum += points->features_array[f][i];
			}
			std::cout << sum << ", ";
		}

		std::cout << std::endl;
		std::cout << "summed with one kernel: ";
		for (int f = 0; f < out->num_features; f++)
		{
			int sum = 0;
			for (int i = 0; i < tmp; i++)
				if (out->cluster_id_of_point[i] == c)
					// sum += out->features_array[f][c + i * centroids->num_data_points];
					sum += out->features_array[f][i];
			std::cout << sum << ", ";
		}

		std::cout << std::endl;
		std::cout << "sumed all with kernel:  ";
		for (int f = 0; f < ret->num_features; f++)
		{
			int sum = 0;
			// for (int i = 0; i < 1; i++)
			sum += ret->features_array[f][c];
			std::cout << sum << ", ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	DeallocateDataPoints(out);
	DeallocateDataPoints(ret);
	cudaCheckError();
}