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
	extern __shared__ MyDataType shm[];
	int tid = threadIdx.x;
	long gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int feature_stirde = k;
	int num_clusters = k;
	int num_features = points->num_features;
	if (gid >= num_data_points)
	{
		return;
	}
	// // shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]
	// shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]
	for (int f = 0; f < points->num_features; ++f)
	{

		for (int c = 0; c < num_clusters; ++c)
		{
			// shm[((f * feature_stirde) + c) + tid * points->num_features * k] = 0;
			shm[INDEX(f, c, tid, feature_stirde, k, num_features)] = 0;
			shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] = 0;
		}

		int c = points->cluster_id_of_point[gid];
		// shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid];
		shm[INDEX(f, c, tid, feature_stirde, k, num_features)] += points->features_array[f][gid];

		if (gid + blockDim.x >= num_data_points)
		{
			continue;
		}

		c = points->cluster_id_of_point[gid + blockDim.x];
		// shm[((f * feature_stirde) + c) + tid * points->num_features * k] += points->features_array[f][gid + blockDim.x];
		shm[INDEX(f, c, tid, feature_stirde, k, num_features)] += points->features_array[f][gid + blockDim.x];
		// shm[INDEX_ID(c, tid, feature_stirde, k, points->num_features)] += 1;
		// idx where to store particualr feature coord
	}
	{
		int c = points->cluster_id_of_point[gid];
		if (count_in)
			shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] += count_in;
		else
			shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] += count_out[gid];

		if (gid + blockDim.x < num_data_points)
		{
			c = points->cluster_id_of_point[gid + blockDim.x];
			if (count_in)
				shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] += count_in;
			else
				shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] += count_out[gid + blockDim.x];
		}
	}
	// else
	// 	return;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		// problem jest gdy num_threads nie jest wiekokrotnoscia 2, wtedy jak mamy blockdim.x/2 itd, dostanimy cos co sie nie podzieli przez 2
		if (tid < stride)
		{
			for (int f = 0; f < num_features; ++f)
			{
				for (int c = 0; c < num_clusters; ++c)
				{
					// czy to jest optumalny odczyt??
					// shm[((f * feature_stirde) + c) + tid * points->num_features * k] += shm[((f * feature_stirde) + c) + (tid + stride) * points->num_features * k];
					shm[INDEX(f, c, tid, feature_stirde, k, num_features)] += shm[INDEX(f, c, (tid + stride), feature_stirde, k, num_features)];
				}
			}
			for (int c = 0; c < num_clusters; ++c)
			{
				shm[INDEX_ID(c, tid, feature_stirde, k, num_features)] += shm[INDEX_ID(c, (tid + stride), feature_stirde, k, num_features)];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int f = 0; f < num_features; ++f)
		{
			for (int c = 0; c < num_clusters; ++c)
			{
				// out->features_array[f][c + blockIdx.x * k] = shm[((f * feature_stirde) + c)];
				out->features_array[f][c + blockIdx.x * k] = shm[INDEX(f, c, 0, feature_stirde, k, num_features)];
			}
		}
		for (int c = 0; c < num_clusters; ++c)
		{
			// [{count1,...,count5},{count1,...,count5},..,
			count_out[blockIdx.x * k + c] = shm[INDEX_ID(c, 0, feature_stirde, k, num_features)];
		}
	}
}

__global__ void FindNewCentroids(DataPoints *centroids, int *count, DataPoints *reduced_points)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int f = gid / centroids->num_data_points;
	int c = gid % centroids->num_data_points;
	// if (gid >= centroids->num_features * centroids->num_data_points)
	// {
	// 	return;
	// }
	centroids->features_array[f][c] = reduced_points->features_array[f][c] / (float)count[c];
}

__global__ void InitPointsWithCentroidsIds(DataPoints *points, int k, int num_points)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= num_points)
	{
		return;
	}
	points->cluster_id_of_point[gid] = gid % k;
}
#define DEBUG 0
static int ep = 0;
static float **copy;
void KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{
// std::cout<<"asfs  "<<sizeof(**(points->features_array))<<"  \n";
// std::cout<<"asfs  "<<typedef(**(points->features_array))<<"  \n";
	const int num_features = points->num_features;
	const int num_clusters = centroids->num_data_points;
	int N = points->num_data_points;
	const int num_threads = 1024 / 4;
	int num_blocks = (int)std::max(std::ceil((int)(N / (double)num_threads)), 1.0);
	size_t shmem_size = num_threads * sizeof(MyDataType) * num_features * num_clusters + num_threads * sizeof(MyDataType) * num_clusters;
	cudaDeviceSynchronize();
	DataPoints *debug = AllocateDataPoints(num_features, num_clusters);
	double sum_tot = 0;
	cudaDeviceSynchronize();

	// timer_find_closest_centroids->Start();
	// if (ep >0)
	if (ep > 0)
	{
		points->num_features = num_features;
		cudaMallocManaged(&(copy), sizeof(float *) * points->num_features);
		cudaCheckError();

		for (int feature = 0; feature < points->num_features; ++feature)
		{
			cudaMallocManaged(&(copy[feature]), sizeof(*(copy[feature])) * points->num_data_points);
			cudaCheckError();
			// cudaMemcpy(copy[feature], points->features_array[feature], sizeof(*(copy[feature])) * points->num_data_points, cudaMemcpyKind::cudaMemcpyDefault);
			// cudaCheckError();
		}
		double check = 0;
		double checkv2 = 0;
		for (int i = 0; i < points->num_data_points; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				copy[f][i] = points->features_array[f][i];
				check += copy[f][i];
				checkv2 += points->features_array[f][i];
			}
		}
		std::cout << "check: " << check << std::endl;
		;
		std::cout << "check2: " << checkv2 << std::endl;
		;
	}
	ep--;
	cudaDeviceSynchronize();
	FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);

	// timer_find_closest_centroids->Stop();
	// timer_find_closest_centroids->Elapsed();
	cudaDeviceSynchronize();

	cudaCheckError();
	// if (DEBUG)
	// {
	// 	int *exact_num = new int[num_clusters];
	// 	for (int i = 0; i < num_clusters; i++)
	// 	{
	// 		exact_num[i] = 0;
	// 	}
	// 	for (int i = 0; i < N; i++)
	// 	{
	// 		exact_num[points->cluster_id_of_point[i]]++;
	// 	}
	// 	std::cout << std::endl;
	// 	int sum_tot = 0;
	// 	for (int i = 0; i < num_clusters; i++)
	// 	{
	// 		sum_tot += exact_num[i];
	// 		std::cout << exact_num[i] << ", ";
	// 	}
	// 	free(exact_num);
	// 	std::cout << std::endl
	// 			  << N << " total sum: " << sum_tot << std::endl;
	// }
	num_blocks = std::ceil(num_blocks / 2.0);
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
	cudaDeviceSynchronize();
	DataPoints *out = AllocateDataPoints(num_features, tmp);
	cudaDeviceSynchronize();
	const int num_threads_inti_id = std::min(1024, tmp);
	const int num_block_init_id = (int)std::max(std::ceil((int)(tmp / (double)num_threads_inti_id)), 1.0);
	InitPointsWithCentroidsIds<<<num_block_init_id, num_threads_inti_id>>>(out, num_clusters, tmp);
	cudaDeviceSynchronize();
	cudaCheckError();

	int *count_out;
	cudaMallocManaged(&count_out, sizeof(int) * num_blocks * num_clusters);
	cudaCheckError();
	cudaMemset(count_out, 0, sizeof(int) * num_blocks * num_clusters);
	cudaCheckError();
	if (DEBUG)
	{
		for (int f = 0; f < num_features; f++)
			for (int c = 0; c < num_clusters; c++)
			{
				debug->features_array[f][c] = 0;
			}
		sum_tot = 0;

		for (int i = 0; i < points->num_data_points; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				debug->features_array[f][points->cluster_id_of_point[i]] += points->features_array[f][i];
				sum_tot += points->features_array[f][i];
			}
		}
		double cc[2][3];
		for (int c = 0; c < num_clusters; c++)
			for (int f = 0; f < num_features; f++){
				cc[f][c]=0;
			}
		for (int i = 0; i < points->num_data_points; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				cc[f][points->cluster_id_of_point[i]] += points->features_array[f][i];
				// sum_tot += points->features_array[f][i];
			}
		}

		for (int c = 0; c < num_clusters; c++)
			for (int f = 0; f < num_features; f++)
			{
				std::cout << cc[f][c] << ", ";
			}
		std::cout << " correct\n";
		double sum_tot_v2 = 0;
		for (int c = 0; c < num_clusters; c++)
			for (int f = 0; f < num_features; f++)
			{
				std::cout << debug->features_array[f][c] << ", ";
				sum_tot_v2 += debug->features_array[f][c];
				debug->features_array[f][c] = 0;
			}
		std::cout << "sumed all points: " << sum_tot << std::endl;
		std::cout << "sumed all points vv22: " << sum_tot_v2 << std::endl;
	}

	if (DEBUG)
		if (num_blocks * num_threads * 2 < N || out->num_data_points != num_clusters * num_blocks)
		{
			std::cout << "11aaaaaaaaaaaaaaaaaaaaaa\n";
		}
	cudaDeviceSynchronize();
	// timer_compute_centroids->Start();
	ReduceDataPoints<<<num_blocks, num_threads, shmem_size>>>(points, num_clusters, out, 1, count_out, N);
	// czasem mu sie coś wali w obliczeniach, np. dla n=1<<22
	// jakby nie sumował wszyskiego czasem
	// timer_compute_centroids->Stop();
	// timer_compute_centroids->Elapsed();
	cudaDeviceSynchronize();

	cudaCheckError();
	if (DEBUG)
	{
		sum_tot = 0;
		for (int i = 0; i < num_clusters * num_blocks; i++)
		{
			sum_tot += count_out[i];
			// std::cout << "coutout: " << count_out[i] << ",  ";
		}
		std::cout << "tot_sum " << sum_tot << " N: " << points->num_data_points << std::endl;

		for (int i = 0; i < num_clusters * num_blocks; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				debug->features_array[f][out->cluster_id_of_point[i]] += out->features_array[f][i];
			}
		}

		sum_tot = 0;
		for (int f = 0; f < num_features; f++)
			for (int c = 0; c < num_clusters; c++)
			{
				std::cout << debug->features_array[f][c] << ", ";
				sum_tot += debug->features_array[f][c];
				debug->features_array[f][c] = 0;
			}
		std::cout << "sumed all points: " << sum_tot << std::endl;
	}

	N = num_blocks * num_clusters;
	// N=lambda(N);
	const int new_num_block = std::ceil(N / num_threads / 2.0);
	shmem_size = num_threads * sizeof(points->features_array[0][0]) * num_features * num_clusters + num_threads * sizeof(points->features_array[0][0]) * num_clusters;
	if (DEBUG)
		if (new_num_block * num_threads * 2 < N)
		{
			std::cout << "222aaaaaaaaaaaaaaaaaaaaaa\n";
		}
	cudaDeviceSynchronize();
	ReduceDataPoints<<<new_num_block, num_threads, shmem_size>>>(out, num_clusters, out, 0, count_out, N);
	cudaDeviceSynchronize();
	cudaCheckError();
	if (DEBUG)
	{
		sum_tot = 0;
		for (int i = 0; i < new_num_block * num_clusters; i++)
		{
			sum_tot += count_out[i];
			// std::cout << "coutout: " << count_out[i] << ",  ";
		}
		std::cout << "tot_sum " << sum_tot << " N: " << points->num_data_points << std::endl;

		for (int i = 0; i < num_clusters * new_num_block; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				debug->features_array[f][out->cluster_id_of_point[i]] += out->features_array[f][i];
			}
		}
		sum_tot = 0;
		for (int f = 0; f < num_features; f++)
			for (int c = 0; c < num_clusters; c++)
			{
				std::cout << debug->features_array[f][c] << ", ";
				sum_tot += debug->features_array[f][c];
				debug->features_array[f][c] = 0;
			}
		std::cout << "sumed all points: " << sum_tot << std::endl;
	}
	cudaDeviceSynchronize();
	// timer_compute_centroids->Start();
	N = num_clusters * new_num_block;
	int num_threads_last_sumup = std::ceil(N / 2.0);
	num_threads_last_sumup = lambda(num_threads_last_sumup);
	if (new_num_block > 1)
	{
		if (DEBUG)
			if (1 * 2 * num_threads_last_sumup < N)
			{
				std::cout << "333aaaaaaaaaaaaaaaaaaaaaa\n";
			}
		shmem_size = num_threads_last_sumup * sizeof(float) * num_features * num_clusters + num_threads_last_sumup * sizeof(float) * num_clusters;
		ReduceDataPoints<<<1, num_threads_last_sumup, shmem_size>>>(out, num_clusters, out, 0, count_out, N);
	}
	cudaDeviceSynchronize();

	// timer_compute_centroids->Stop();
	// timer_compute_centroids->Elapsed();
	cudaCheckError();
	cudaDeviceSynchronize();
	// FindNewCentroids<<<1, num_features * num_clusters>>>(centroids, count_out, out);
	// cudaDeviceSynchronize();
	for (int f = 0; f < num_features; f++)
		for (int c = 0; c < num_clusters; c++)
		{

			centroids->features_array[f][c] = out->features_array[f][c] / (double)count_out[c];
		}
	cudaDeviceSynchronize();
	cudaCheckError();
	if (DEBUG)
	{
		sum_tot = 0;
		for (int i = 0; i < num_clusters * 1; i++)
		{
			sum_tot += count_out[i];
			// std::cout << "coutout: " << count_out[i] << ",  ";
		}
		std::cout << "tot_sum " << sum_tot << " N: " << points->num_data_points << std::endl;

		for (int i = 0; i < num_clusters * 1; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				debug->features_array[f][out->cluster_id_of_point[i]] += out->features_array[f][i];
			}
		}
		sum_tot = 0;
		for (int f = 0; f < num_features; f++)
			for (int c = 0; c < num_clusters; c++)
			{
				std::cout << debug->features_array[f][c] << ", ";
				sum_tot += debug->features_array[f][c];
				debug->features_array[f][c] = 0;
			}
		std::cout << "sumed all points: " << sum_tot << std::endl;

		for (int i = 0; i < points->num_data_points; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				debug->features_array[f][points->cluster_id_of_point[i]] += points->features_array[f][i];
			}
		}
		sum_tot = 0;
		for (int f = 0; f < num_features; f++)
			for (int c = 0; c < num_clusters; c++)
			{
				std::cout << debug->features_array[f][c] << ", ";
				sum_tot += debug->features_array[f][c];
				debug->features_array[f][c] = 0;
			}
		std::cout << "sumed all points: " << sum_tot << std::endl;

		int *count_check = (int *)malloc(sizeof(int) * num_clusters);
		memset(count_check, 0, sizeof(int) * num_clusters);
		for (int i = 0; i < points->num_data_points; i++)
		{
			count_check[points->cluster_id_of_point[i]]++;
		}
		std::cout << "exact_count: ";
		for (int c = 0; c < num_clusters; c++)
		{
			std::cout << count_check[c] << ", ";
		}
		std::cout << std::endl
				  << "count_out:   ";
		for (int c = 0; c < num_clusters; c++)
		{
			std::cout << count_out[c] << ", ";
		}
		std::cout << std::endl;

		std::cout << std::endl;
		std::cout << std::endl;
		free(count_check);
	}
	// int allsum = 0;
	// for (int i = 0; i < points->num_data_points; i++)
	// {
	// 	for (int f = 0; f < num_features; f++)
	// 	{
	// 		if (points->features_array[f][i] != copy[f][i])
	// 			std::cout << points->features_array[f][i] << ", ----------------------";
	// 		allsum += copy[f][i];
	// 	}
	// }
	// std::cout << "copy sum: " << allsum << std::endl;
	// std::cout << std::endl;
	cudaDeviceSynchronize();
	DeallocateDataPoints(out);
	DeallocateDataPoints(debug);
	cudaFree(count_out);
	cudaCheckError();
}