#include "kMeansGpu.h"
#include "findClosestCentriods.h"
#include "cuda.h"
#include "cudaCheckError.h"
#include "timer.h"
#include "Constants.h"

#define INDEX(f, c, tid, num_clusters, num_features) ((f * num_clusters) + c) + tid *(num_features + 1) * num_clusters
#define INDEX_ID(c, tid, num_clusters, num_features) ((num_features * num_clusters) + c) + tid *(num_features + 1) * num_clusters

// template <int NUM_FEATURES=2,int NUM_DATA_POINTS=200>
// __global__ void ReduceDataPoints(const DataPoints *points, const int k /*number of centroids*/, DataPoints *out, const int count_in, long *count_out, const int num_data_points)
__global__ void ReduceDataPoints(MyDataType **features, int *cluster_ids, MyDataType **centroids_features,
								 const int count_in, long *count_out, const int num_data_points, const int num_features, const int num_clusters)
{
	extern __shared__ MyDataType shm[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// jakbybyły tempalte to można by trochę obliczeń zrobić w czasie kompilacji głownie indexy
	if (gid >= num_data_points)
	{
		return;
	}
	// // shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5}) ]
	// shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}) ]

	int c1, c2;

	for (int f = 0; f < num_features; ++f)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			shm[INDEX(f, c, tid, num_clusters, num_features)] = 0;
			shm[INDEX_ID(c, tid, num_clusters, num_features)] = 0;
		}

		c1 = cluster_ids[gid];
		shm[INDEX(f, c1, tid, num_clusters, num_features)] += features[f][gid];

		// problem: if datatype is long double then c ==-1 because in genertepoint  i set its clusterid to -1, but why onyl when long double?? ok its because shared memory is long double i gesss

		if (gid + blockDim.x >= num_data_points)
		{
			continue;
		}

		c2 = cluster_ids[gid + blockDim.x];
		shm[INDEX(f, c2, tid, num_clusters, num_features)] += features[f][gid + blockDim.x];
		// idx where to store particualr feature coord
	}
	{
		// int c = cluster_ids[gid];
		if (count_in)
			shm[INDEX_ID(c1, tid, num_clusters, num_features)] = count_in;
		else
			shm[INDEX_ID(c1, tid, num_clusters, num_features)] = count_out[gid];

		if (gid + blockDim.x < num_data_points)
		{
			if (count_in)
				shm[INDEX_ID(c2, tid, num_clusters, num_features)] += count_in;
			else
				shm[INDEX_ID(c2, tid, num_clusters, num_features)] += count_out[gid + blockDim.x];
		}
	}
	// else
	// 	return;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
		{
			for (int f = 0; f < num_features; ++f)
			{
				for (int c = 0; c < num_clusters; ++c)
				{
					// czy to jest optumalny odczyt??
					shm[INDEX(f, c, tid, num_clusters, num_features)] += shm[INDEX(f, c, (tid + stride), num_clusters, num_features)];
					if (f == 0)
					{
						shm[INDEX_ID(c, tid, num_clusters, num_features)] += shm[INDEX_ID(c, (tid + stride), num_clusters, num_features)];
					}
				}
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int f = 0; f < num_features; ++f)
			for (int c = 0; c < num_clusters; ++c)
			{
				{
					centroids_features[f][c + blockIdx.x * num_clusters] = shm[INDEX(f, c, 0, num_clusters, num_features)];
					if (f == 0)
					{
						// [{count1,...,count5},{count1,...,count5},..,
						count_out[blockIdx.x * num_clusters + c] = shm[INDEX_ID(c, 0, num_clusters, num_features)];
					}
				}
			}
	}
}

__global__ void FindNewCentroids(DataPoints *centroids, long *count, DataPoints *reduced_points)
{
	// może zrobić to na talbicy wątkow dwuwymiarowej??
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int f = gid / centroids->num_data_points;
	const int c = gid % centroids->num_data_points;
	if (gid >= centroids->num_features * centroids->num_data_points)
	{
		return;
	}
	centroids->features_array[f][c] = reduced_points->features_array[f][c] / (MyDataType)count[c];
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
#define MAX_SHM_SIZE 48 * 1024
#define DEFAULT_NUM_THREADS 1024l
#define CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads) num_threads *(num_features + 1) * num_clusters * sizeof(MyDataType)

DataPoints *reduced_points;
long *ids_count;
int cur_epoch = 0;




void KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{
	const int num_features = points->num_features;
	const int num_clusters = centroids->num_data_points;
	int N = points->num_data_points;
	int num_threads = DEFAULT_NUM_THREADS;

	while (MAX_SHM_SIZE < CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads))
	{
		num_threads /= 2;
	}

	long num_blocks = (int)std::max(std::ceil((long)(N / (double)num_threads)), 1.0);
	size_t shmem_size = CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads);

	DataPoints *debug;
	if (DEBUG)
	{
		debug = AllocateDataPoints(num_features, num_clusters);
	}

	// Find closest centroids for each datapoint
	const int num_threads_find_closest = 1024;
	const int num_blocks_find_closest = std::max(1, (int)std::ceil(points->num_data_points / num_threads_find_closest));
	const size_t shm_find_closest = sizeof(MyDataType) * num_clusters * num_features + sizeof(MyDataType) * num_threads_find_closest * num_features;
	timer_find_closest_centroids->Start();
	FindClosestCentroids<<<num_blocks_find_closest, num_threads_find_closest, shm_find_closest>>>(points->features_array,
																								  points->cluster_id_of_point, centroids->features_array, points->num_data_points,
																								  num_features, num_clusters);
	timer_find_closest_centroids->Stop();
	timer_find_closest_centroids->Elapsed();
	cudaCheckError();
	// Find closest centroids for each datapoint

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

	// Create and init reduced points, what will be used sum up all points
	num_blocks = std::ceil(num_blocks / 2.0);
	// tmp = lambda(tmp);
	long num_reduced_points = num_blocks * num_clusters;
	if (cur_epoch == 0)
	{
		reduced_points = AllocateDataPoints(num_features, num_reduced_points);
		int num_threads_inti_id = (int)std::min(DEFAULT_NUM_THREADS, num_reduced_points);
		int num_block_init_id = (int)std::max(std::ceil((num_reduced_points / (double)num_threads_inti_id)), 1.0);
		InitPointsWithCentroidsIds<<<num_block_init_id, num_threads_inti_id>>>(reduced_points, num_clusters, num_reduced_points);
		cudaDeviceSynchronize();
		cudaCheckError();
		cudaMallocManaged(&ids_count, sizeof(long) * num_blocks * num_clusters);
		cudaCheckError();
		cudaMemset(ids_count, 0, sizeof(long) * num_blocks * num_clusters);
		cudaCheckError();
	}
	else
	{
		// clean recued_points??
	}
	// Create and init reduced points

	if (DEBUG)
	{
		long sum_tot = 0;
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
			for (int f = 0; f < num_features; f++)
			{
				cc[f][c] = 0;
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
		if (num_blocks * num_threads * 2 < N || reduced_points->num_data_points != num_clusters * num_blocks)
		{
			std::cout << "11aaaaaaaaaaaaaaaaaaaaaa\n";
		}
	}
	// reduce points in `points` and store them in `reduced_poitsn`
	timer_compute_centroids->Start();
	ReduceDataPoints<<<num_blocks, num_threads, shmem_size>>>(points->features_array,
															  points->cluster_id_of_point, reduced_points->features_array,
															  1, ids_count, N, num_features, num_clusters);
	timer_compute_centroids->Stop();
	timer_compute_centroids->Elapsed();
	cudaCheckError();
	// reduce points in `points` and store them in reduced_poitsn

	// further reduce points in `reduced_points`, until there will be no more then  `num_threads * 2` poitns left to reduce
	while (num_blocks * num_clusters > num_threads * 2)
	{

		if (DEBUG)
		{
			long sum_tot = 0;
			sum_tot = 0;
			for (int i = 0; i < num_clusters * num_blocks; i++)
			{
				sum_tot += ids_count[i];
			}
			std::cout << "tot_sum " << sum_tot << " N: " << points->num_data_points << std::endl;

			for (int i = 0; i < num_clusters * num_blocks; i++)
			{
				for (int f = 0; f < num_features; f++)
				{
					debug->features_array[f][reduced_points->cluster_id_of_point[i]] += reduced_points->features_array[f][i];
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
			if (num_blocks * num_threads * 2 < N)
			{
				std::cout << "222aaaaaaaaaaaaaaaaaaaaaa\n";
			}
		}
		N = num_blocks * num_clusters;
		// N=lambda(N);
		num_blocks = std::ceil(N / num_threads / 2.0);
		shmem_size = CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads);

		cudaDeviceSynchronize();
		timer_compute_centroids->Start();
		ReduceDataPoints<<<num_blocks, num_threads, shmem_size>>>(reduced_points->features_array,
																  reduced_points->cluster_id_of_point, reduced_points->features_array,
																  0, ids_count, N, num_features, num_clusters);
		timer_compute_centroids->Stop();
		timer_compute_centroids->Elapsed();
		cudaCheckError();
	}
	// further reduce points in `reduced_points`, until there will be no more then  `num_threads * 2` poitns left to reduce

	// last reduce, reduce all remaining points to a 'single datapoint', that is: points belonging to the same cluster will be reduced to single point
	if (num_blocks > 1) // if may happen, that last reduced reduced all the point, happen when num_blocks==1
	{
		N = num_clusters * num_blocks;
		int num_threads_last_sumup = std::ceil(N / 2.0);
		num_threads_last_sumup = lambda(num_threads_last_sumup);
		if (DEBUG)
		{
			long sum_tot = 0;
			sum_tot = 0;
			for (int i = 0; i < num_blocks * num_clusters; i++)
			{
				sum_tot += ids_count[i];
				// std::cout << "coutout: " << count_out[i] << ",  ";
			}
			std::cout << "tot_sum " << sum_tot << " N: " << points->num_data_points << std::endl;

			for (int i = 0; i < num_clusters * num_blocks; i++)
			{
				for (int f = 0; f < num_features; f++)
				{
					debug->features_array[f][reduced_points->cluster_id_of_point[i]] += reduced_points->features_array[f][i];
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
			if (1 * 2 * num_threads_last_sumup < N)
			{
				std::cout << "333aaaaaaaaaaaaaaaaaaaaaa\n";
			}
		}
		shmem_size = CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads_last_sumup);
		timer_compute_centroids->Start();
		ReduceDataPoints<<<1, num_threads_last_sumup, shmem_size>>>(reduced_points->features_array,
																	reduced_points->cluster_id_of_point, reduced_points->features_array,
																	0, ids_count, N, num_features, num_clusters);
		timer_compute_centroids->Stop();
		timer_compute_centroids->Elapsed();
		cudaCheckError();
	}
	// last reduce, reduce all remaining points

	// find new centroids
	FindNewCentroids<<<1, num_features * num_clusters>>>(centroids, ids_count, reduced_points);
	cudaDeviceSynchronize();
	cudaCheckError();
	// find new centroids

	if (DEBUG)
	{
		long sum_tot = 0;
		sum_tot = 0;
		for (int i = 0; i < num_clusters * 1; i++)
		{
			sum_tot += ids_count[i];
			// std::cout << "coutout: " << count_out[i] << ",  ";
		}
		std::cout << "tot_sum " << sum_tot << " N: " << points->num_data_points << std::endl;

		for (int i = 0; i < num_clusters * 1; i++)
		{
			for (int f = 0; f < num_features; f++)
			{
				debug->features_array[f][reduced_points->cluster_id_of_point[i]] += reduced_points->features_array[f][i];
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
			std::cout << ids_count[c] << ", ";
		}
		std::cout << std::endl;

		std::cout << std::endl;
		std::cout << std::endl;
		free(count_check);
	}

	// cleanup memory
	if (cur_epoch == constants::num_epoches - 1)
	{
		DeallocateDataPoints(reduced_points);
		cudaFree(ids_count);
	}
	if (DEBUG)
	{
		DeallocateDataPoints(debug);
	}
	cudaCheckError();
}