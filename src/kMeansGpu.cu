#include "kMeansGpu.h"
#include "findClosestCentriods.h"
#include "cuda.h"
#include "cudaCheckError.h"
#include "timer.h"

#define INDEX(f, c, tid, feature_stide, k, num_features) ((f * feature_stirde) + c) + tid *(num_features + 1) * k
#define INDEX_ID(c, tid, feature_stide, k, num_features) ((num_features * feature_stirde) + c) + tid *(num_features + 1) * k

// template <int NUM_FEATURES=2,int NUM_DATA_POINTS=200>
__global__ void ReduceDataPoints(const DataPoints *points, int k /*number of centroids*/, DataPoints *out, int count_in, int *count_out, int num_data_points)
{
	extern __shared__ float shm[];
	int tid = threadIdx.x;
	int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int feature_stirde = k;
	int num_clusters = k;
	int num_features = points->num_features;

	// // shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]
	// shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]
	for (int f = 0; f < points->num_features; ++f)
	{

		for (int c = 0; c < k; ++c)
		{
			// shm[((f * feature_stirde) + c) + tid * points->num_features * k] = 0;
			shm[INDEX(f, c, tid, feature_stirde, k, num_features)] = 0;
			shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] = 0;
		}
		if (gid + blockDim.x >= num_data_points)
		{
			break;
		}

		int c = points->cluster_id_of_point[gid];

		// shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid];
		shm[INDEX(f, c, tid, feature_stirde, k, points->num_features)] += points->features_array[f][gid];

		c = points->cluster_id_of_point[gid + blockDim.x];
		// shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid + blockDim.x];
		shm[INDEX(f, c, tid, feature_stirde, k, points->num_features)] += points->features_array[f][gid + blockDim.x];
		// shm[INDEX_ID(c, tid, feature_stirde, k, points->num_features)] += 1;
		// idx where to store particualr feature coord
	}

	int c = points->cluster_id_of_point[gid];
	if (count_in)
		shm[INDEX_ID(c, tid, feature_stirde, k, points->num_features)] = count_in;
	else
		shm[INDEX_ID(c, tid, feature_stirde, k, points->num_features)] = count_out[gid];

	c = points->cluster_id_of_point[gid + blockDim.x];
	if (count_in)
		shm[INDEX_ID(c, tid, feature_stirde, k, points->num_features)] += count_in;
	else
		shm[INDEX_ID(c, tid, feature_stirde, k, points->num_features)] += count_out[gid + blockDim.x];

	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		// problem jest gdy num_threads nie jest wiekokrotnoscia 2, wtedy jak mamy blockdim.x/2 itd, dostanimy cos co sie nie podzieli przez 2
		if (tid < stride)
		{
			for (int f = 0; f < num_features; ++f)
			{
				for (int c = 0; c < k; ++c)
				{
					// czy to jest optumalny odczyt??
					// shm[((f * feature_stirde) + c) + tid * points->num_features * k] += shm[((f * feature_stirde) + c) + (tid + stride) * points->num_features * k];
					shm[INDEX(f, c, tid, feature_stirde, k, num_features)] += shm[INDEX(f, c, (tid + stride), feature_stirde, k, num_features)];
				}
			}
			for (int c = 0; c < k; ++c)
			{
				shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] += shm[INDEX_ID(c, (tid + stride), feature_stirde, k, num_features)];
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
				// out->features_array[f][c + blockIdx.x * k] = shm[((f * feature_stirde) + c)];
				out->features_array[f][c + blockIdx.x * k] = shm[INDEX(f, c, 0, feature_stirde, k, num_features)];
			}
		}
		for (int c = 0; c < k; ++c)
		{
			// [{count1,...,count5},{count1,...,count5},..,
			count_out[blockIdx.x * k + c] = shm[INDEX_ID(c, 0, feature_stirde, k, num_features)];
		}
	}
}

__global__ void FindNewCentroids(DataPoints *centroids, int *count, DataPoints *reduced_points)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int f = gid / centroids->num_data_points;
	int c = gid % centroids->num_data_points;
	centroids->features_array[f][c] = reduced_points->features_array[f][c] / (float)count[c];
}

__global__ void InitPointsWithCentroidsIds(DataPoints *points, int k)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	points->cluster_id_of_point[gid] = gid % k;
}

#include <iostream>
void KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{
	int num_features = points->num_features;
	int num_clusters = centroids->num_data_points;
	int N = points->num_data_points;
	int num_threads = 1024;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	size_t shmem_size = num_threads * sizeof(float) * num_features * num_clusters + num_threads * sizeof(float) * num_clusters;

	timer_find_closest_centroids.Start();
	FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);
	timer_find_closest_centroids.Stop();
	timer_find_closest_centroids.Elapsed();
	cudaCheckError();

	num_blocks = std::ceil(num_blocks / 2);
	// rewerite lambda
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
	int tmp = num_blocks * num_clusters;
	tmp = lambda(tmp);
	DataPoints *out = AllocateDataPoints(num_features, tmp);
	int nt = (int)std::max(std::ceil((int)(tmp / num_threads)), 1.0);
	InitPointsWithCentroidsIds<<<nt, num_threads>>>(out, num_clusters);
	cudaDeviceSynchronize();
	cudaCheckError();

	int *count_out;
	cudaMallocManaged(&count_out, sizeof(int) * num_blocks * num_clusters);
	cudaCheckError();
	cudaMemset(count_out, 0, sizeof(int) * num_blocks * num_clusters);
	cudaCheckError();

	while (1)
	{

		break;
	}

	// second reduce
	timer_compute_centroids.Start();
	// cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	ReduceDataPoints<<<num_blocks, num_threads, shmem_size>>>(points, num_clusters, out, 1, count_out, points->num_data_points);
	timer_compute_centroids.Stop();
	timer_compute_centroids.Elapsed();
	cudaCheckError();
	// second reduce

	num_threads = tmp / 2;
	shmem_size = num_threads * sizeof(float) * num_features * num_clusters + num_threads * sizeof(float) * num_clusters;

	timer_compute_centroids.Start();
	ReduceDataPoints<<<1, num_threads, shmem_size>>>(out, num_clusters, out, 0, count_out, out->num_data_points);
	timer_compute_centroids.Stop();
	timer_compute_centroids.Elapsed();
	cudaCheckError();
	FindNewCentroids<<<1, num_features * num_clusters>>>(centroids, count_out, out);
	cudaDeviceSynchronize();

	// for (int f = 0; f < num_features; f++)
	// 	for (int c = 0; c < num_clusters; c++)
	// 		centroids->features_array [f][c] = out->features_array[f][c] / count_out[c];

	DeallocateDataPoints(out);
	cudaFree(count_out);
	cudaCheckError();
}