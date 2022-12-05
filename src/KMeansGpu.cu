#include "KMeansGpu.h"

#include "cuda.h"

#include "Config.h"
#include "CudaCheckError.h"
#include "FindClosestCentriods.h"
#include "Timer.h"

// kernels

#define INDEX(f, c, tid, num_clusters, num_features) ((f * num_clusters) + c) + tid *(num_features + 1) * num_clusters
#define INDEX_ID(c, tid, num_clusters, num_features) ((num_features * num_clusters) + c) + tid *(num_features + 1) * num_clusters
template <int F_NUM>
__global__ void ReduceDataPoints(MyDataType **features, int *cluster_ids, MyDataType **centroids_features,
								 const int count_in, CountType *count_out, const int num_data_points, const int num_clusters, int act)
{
	extern __shared__ MyDataType shm[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (gid >= num_data_points)
	{
		return;
	}
	// shm[(f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}), (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}),..., (f1{c1,c2,c3,c4,c5},f2{c1,c2,c3,c4,c5},{count1,...,count5}) ]

	int c1, c2;

	for (int f = 0; f < F_NUM; ++f)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			shm[INDEX(f, c, tid, num_clusters, F_NUM)] = 0;
			shm[INDEX_ID(c, tid, num_clusters, F_NUM)] = 0;
		}

		c1 = cluster_ids[gid];
		shm[INDEX(f, c1, tid, num_clusters, F_NUM)] += features[f][gid];

		if (gid + blockDim.x >= num_data_points)
		{
			continue;
		}

		c2 = cluster_ids[gid + blockDim.x];
		shm[INDEX(f, c2, tid, num_clusters, F_NUM)] += features[f][gid + blockDim.x];
		// idx where to store particualr feature coord
	}
	{
		if (count_in)
			shm[INDEX_ID(c1, tid, num_clusters, F_NUM)] = count_in;
		else
			shm[INDEX_ID(c1, tid, num_clusters, F_NUM)] = count_out[gid];

		if (gid + blockDim.x < num_data_points)
		{
			if (count_in)
				shm[INDEX_ID(c2, tid, num_clusters, F_NUM)] += count_in;
			else
				shm[INDEX_ID(c2, tid, num_clusters, F_NUM)] += count_out[gid + blockDim.x];
		}
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
		{
			for (int f = 0; f < F_NUM; ++f)
			{
				for (int c = 0; c < num_clusters; ++c)
				{
					shm[INDEX(f, c, tid, num_clusters, F_NUM)] += shm[INDEX(f, c, (tid + stride), num_clusters, F_NUM)];
					if (f == 0)
					{
						shm[INDEX_ID(c, tid, num_clusters, F_NUM)] += shm[INDEX_ID(c, (tid + stride), num_clusters, F_NUM)];
					}
				}
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int f = 0; f < F_NUM; ++f)
			for (int c = 0; c < num_clusters; ++c)
			{
				{
					centroids_features[f][c + blockIdx.x * num_clusters] = shm[INDEX(f, c, 0, num_clusters, F_NUM)];
					if (f == 0)
					{
						// [{count1,...,count5},{count1,...,count5},..,
						count_out[blockIdx.x * num_clusters + c] = shm[INDEX_ID(c, 0, num_clusters, F_NUM)];
					}
				}
			}
	}
}

#define INDEX_C(c, tid, num_clusters) c + (tid * num_clusters)
__device__ void warpReduceCount(volatile CountType *shm, int tid, int c, int num_clusters)
{
	shm[INDEX_C(c, tid, num_clusters)] += shm[INDEX_C(c, (tid + 32), num_clusters)];
	shm[INDEX_C(c, tid, num_clusters)] += shm[INDEX_C(c, (tid + 16), num_clusters)];
	shm[INDEX_C(c, tid, num_clusters)] += shm[INDEX_C(c, (tid + 8), num_clusters)];
	shm[INDEX_C(c, tid, num_clusters)] += shm[INDEX_C(c, (tid + 4), num_clusters)];
	shm[INDEX_C(c, tid, num_clusters)] += shm[INDEX_C(c, (tid + 2), num_clusters)];
	shm[INDEX_C(c, tid, num_clusters)] += shm[INDEX_C(c, (tid + 1), num_clusters)];
}
__global__ void ReduceDataPointsCountPoints(const int *cluster_ids,
											const CountType count_in, CountType *count_out, const int num_data_points, const int num_clusters, int active_threads_count)
{
	extern __shared__ CountType shm_c[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	int c1, c2;

	for (int c = 0; c < num_clusters; ++c)
	{
		shm_c[INDEX_C(c, tid, num_clusters)] = 0;
	}
	if (gid >= num_data_points)
	{
		return;
	}
	if (tid >= active_threads_count)
	{
		return;
	}
	c1 = cluster_ids[gid];
	if (gid + active_threads_count < num_data_points)
	{
		c2 = cluster_ids[gid + active_threads_count];
	}

	if (count_in)
		shm_c[INDEX_C(c1, tid, num_clusters)] = count_in;
	else
		shm_c[INDEX_C(c1, tid, num_clusters)] = count_out[gid];

	if (gid + active_threads_count < num_data_points)
	{
		if (count_in)
			shm_c[INDEX_C(c2, tid, num_clusters)] += count_in;
		else
			shm_c[INDEX_C(c2, tid, num_clusters)] += count_out[gid + active_threads_count];
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tid < stride)
		{
			for (int c = 0; c < num_clusters; ++c)
			{
				shm_c[INDEX_C(c, tid, num_clusters)] += shm_c[INDEX_C(c, (tid + stride), num_clusters)];
			}
		}
		__syncthreads();
	}
	if (tid < 32)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			warpReduceCount(shm_c, tid, c, num_clusters);
		}
	}
	__syncthreads();

	if (tid == 0)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			count_out[blockIdx.x * num_clusters + c] = shm_c[INDEX_C(c, 0, num_clusters)];
		}
	}
}

#define INDEX_F(c, tid, num_clusters) c + (tid * num_clusters)
__device__ void warpReduceFeature(volatile MyDataType *shm, int tid, int c, int num_clusters)
{
	shm[INDEX_F(c, tid, num_clusters)] += shm[INDEX_F(c, (tid + 32), num_clusters)];
	shm[INDEX_F(c, tid, num_clusters)] += shm[INDEX_F(c, (tid + 16), num_clusters)];
	shm[INDEX_F(c, tid, num_clusters)] += shm[INDEX_F(c, (tid + 8), num_clusters)];
	shm[INDEX_F(c, tid, num_clusters)] += shm[INDEX_F(c, (tid + 4), num_clusters)];
	shm[INDEX_F(c, tid, num_clusters)] += shm[INDEX_F(c, (tid + 2), num_clusters)];
	shm[INDEX_F(c, tid, num_clusters)] += shm[INDEX_F(c, (tid + 1), num_clusters)];
}
__global__ void ReduceDataPointsByFeatures(MyDataType *features, const int *cluster_ids, MyDataType *out,
										   const int num_data_points, const int num_clusters, int active_threads_count)
{
	extern __shared__ MyDataType shm_f[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	int c1, c2;

	for (int c = 0; c < num_clusters; ++c)
	{
		shm_f[INDEX_F(c, tid, num_clusters)] = 0;
	}
	if (gid >= num_data_points)
	{
		return;
	}
	if (tid >= active_threads_count)
	{
		return;
	}
	c1 = cluster_ids[gid];
	if (gid + active_threads_count < num_data_points)
	{
		c2 = cluster_ids[gid + active_threads_count];
	}

	shm_f[INDEX_F(c1, tid, num_clusters)] = features[gid];
	if (gid + active_threads_count < num_data_points)
	{
		shm_f[INDEX_F(c2, tid, num_clusters)] += features[gid + active_threads_count];
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tid < stride)
		{
			for (int c = 0; c < num_clusters; ++c)
			{
				shm_f[INDEX_F(c, tid, num_clusters)] += shm_f[INDEX_F(c, (tid + stride), num_clusters)];
			}
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			warpReduceFeature(shm_f, tid, c, num_clusters);
		}
	}
	__syncthreads();
	if (tid == 0)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			out[c + blockIdx.x * num_clusters] = shm_f[INDEX_F(c, 0, num_clusters)];
		}
	}
}

__global__ void FindNewCentroids(DataPoints *centroids, CountType *count, DataPoints *reduced_points)
{
	const int f = threadIdx.x;
	const int c = threadIdx.y;
	centroids->features_array[f][c] = reduced_points->features_array[f][c] / (MyDataType)count[c];
}

__global__ void InitPointsWithCentroidsIds(DataPoints *points, int num_clusters, int num_points)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= num_points)
	{
		return;
	}
	points->cluster_id_of_point[gid] = gid % num_clusters;
}

// kernels

#define MAX_SHM_SIZE 48 * 1024
#define DEFAULT_NUM_THREADS 1024l
#define CALCULATE_SHM_SIZE_JOIN_REDUCE(num_features, num_clusters, num_threads) num_threads *(num_features + 1) * num_clusters * sizeof(MyDataType)
#define CALCULATE_SHM_SIZE_FEATURE_WISE_REDUCE(num_clusters, num_threads) num_threads *num_clusters * sizeof(MyDataType)
#define CALCULATE_SHM_SIZE_COUNT(num_clusters, num_threads) num_threads *num_clusters * sizeof(CountType)

DataPoints *reduced_points;
CountType *ids_count;
int cur_epoch = 0;

// #define NUM_STREAM (NUM_FEATURES + 1)
// cudaStream_t streams[NUM_STREAM];
template <int F_NUM>
void debugFunction(DataPoints *points, CountType *ids_count, int num_clusters, int num_blocks, int num_threads, int N, std::string label)
{
	std::cout << "\n---------" << label << "---------" << std::endl;
	DataPoints *debug = AllocateDataPoints(F_NUM, num_clusters);

	MyDataType sum_tot = 0;
	// Gets exact sum by feature and clusters
	for (int f = 0; f < F_NUM; f++)
		for (int c = 0; c < num_clusters; c++)
		{
			debug->features_array[f][c] = 0;
		}

	for (int i = 0; i < points->num_data_points; i++)
	{
		for (int f = 0; f < F_NUM; f++)
		{
			debug->features_array[f][points->cluster_id_of_point[i]] += points->features_array[f][i];
			sum_tot += points->features_array[f][i];
		}
	}

	std::cout << " correct (points)\n{\n	";
	double sum_tot_v2 = 0;
	for (int c = 0; c < num_clusters; c++)
	{
		for (int f = 0; f < F_NUM; f++)
		{
			std::cout << debug->features_array[f][c] << ", ";
			sum_tot_v2 += debug->features_array[f][c];
			debug->features_array[f][c] = 0;
		}
	}
	std::cout << "\n}\n";
	// Gets exact sum by feature and clusters

	// Gets redcued sum by feature and cluster
	MyDataType sum_tot_reduced = 0;

	for (int i = 0; i < num_blocks * num_clusters; i++)
	{
		for (int f = 0; f < F_NUM; f++)
		{
			debug->features_array[f][reduced_points->cluster_id_of_point[i]] += reduced_points->features_array[f][i];
			sum_tot_reduced += reduced_points->features_array[f][i];
		}
	}
	std::cout << "Calculated points (reduced_points)\n{\n	";

	double sum_tot_reduced_v2 = 0;
	for (int c = 0; c < num_clusters; c++)
	{
		for (int f = 0; f < F_NUM; f++)
		{
			std::cout << debug->features_array[f][c] << ", ";
			sum_tot_reduced_v2 += debug->features_array[f][c];
			debug->features_array[f][c] = 0;
		}
	}
	std::cout << "\n}\n";

	// Gets redcued sum by feature and cluster

	std::cout << "sumed all points(sum_tot):           " << sum_tot << std::endl;
	std::cout << "sumed all points(sum_tot_v2)         " << sum_tot_v2 << std::endl;
	std::cout << "sumed all points(sum_tot_reduced)    " << sum_tot_reduced << std::endl;
	std::cout << "sumed all points(sum_tot_reduced_v2) " << sum_tot_reduced_v2 << std::endl;

	CountType *count_check = (CountType *)malloc(sizeof(CountType) * num_clusters);
	memset(count_check, 0, sizeof(CountType) * num_clusters);

	// Gets exact count of ids
	CountType exact_points_count = 0;
	for (int i = 0; i < points->num_data_points; ++i)
	{
		count_check[points->cluster_id_of_point[i]]++;
		// std::cout<<points->cluster_id_of_point[i]<<", ";
	}
	std::cout << "Exact ids count\n{\n	";
	for (int c = 0; c < num_clusters; ++c)
	{
		std::cout << count_check[c] << ", ";
		exact_points_count += count_check[c];
	}
	std::cout << "\n}\n";
	// Gets exact count of ids

	memset(count_check, 0, sizeof(CountType) * num_clusters);

	// Gets reduced count of ids
	CountType reduced_points_count = 0;
	for (int i = 0; i < num_blocks; ++i)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			count_check[c] += ids_count[i * num_clusters + c];
			reduced_points_count += ids_count[i * num_clusters + c];
		}
	}
	std::cout << "Reduced ids count\n{\n	";
	for (int c = 0; c < num_clusters; ++c)
	{
		std::cout << count_check[c] << ", ";
	}
	std::cout << "\n}\n";
	// Gets reduced count of ids

	std::cout << "number of points (exact_points_count):   " << exact_points_count << std::endl;
	std::cout << "number of points (reduced_points_count): " << reduced_points_count << std::endl;

	free(count_check);
	if (num_blocks != -1)
		if (num_blocks * num_threads * 2 < N || N != num_clusters * num_blocks)
		{
			std::cout << "aaaaaaaaaaaaaaaaaaaaaa\n";
		}
	DeallocateDataPoints(debug);
	cudaCheckError();
}

int GetNumBlocks(int num_threads, int cur_num_blocks, int num_clusters)
{
	int N = cur_num_blocks * num_clusters;
	int num_blocks = std::ceil((float)N / (float)num_threads / 2.0);
	return num_blocks;
}

template <int F_NUM>
void ReduceFeature(DataPoints *points, DataPoints *out, CountType *ids_count, int num_clusters,
				   int N, CountType count_in, int *num_th, int *num_bl, int atc)
{
	int num_threads = *num_th;
	int num_blocks = *num_bl;
	if (RUN_REDUCE_FEATURE_WISE)
	{
		size_t shm_size = CALCULATE_SHM_SIZE_FEATURE_WISE_REDUCE(num_clusters, num_threads);
		for (int f = 0; f < F_NUM; ++f)
		{
			// ReduceDataPointsByFeatures<<<num_blocks, num_threads, shm_size,streams[f]>>>(points->features_array[f],
			// 																  points->cluster_id_of_point, out->features_array[f],
			// 																  N, num_clusters, atc);
			ReduceDataPointsByFeatures<<<num_blocks, num_threads, shm_size>>>(points->features_array[f],
																			  points->cluster_id_of_point, out->features_array[f],
																			  N, num_clusters, atc);
			cudaCheckError();
		}

		shm_size = CALCULATE_SHM_SIZE_COUNT(num_clusters, num_threads);
		// ReduceDataPointsCountPoints<<<num_blocks, num_threads, shm_size,streams[NUM_STREAM-1]>>>(points->cluster_id_of_point,
		// 																   count_in, ids_count, N, num_clusters, atc);
		ReduceDataPointsCountPoints<<<num_blocks, num_threads, shm_size>>>(points->cluster_id_of_point,
																		   count_in, ids_count, N, num_clusters, atc);
	}
	else if (!RUN_REDUCE_FEATURE_WISE)
	{
		size_t shm_size = CALCULATE_SHM_SIZE_JOIN_REDUCE(F_NUM, num_clusters, num_threads);
		ReduceDataPoints<F_NUM><<<num_blocks, num_threads, shm_size>>>(points->features_array, points->cluster_id_of_point, reduced_points->features_array, count_in, ids_count, N, num_clusters, atc);
	}
}

template <int N_FEATURES>
MyDataType KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{

	// for (int i = 0; i < NUM_STREAM; i++)
	// {
	// 	cudaStreamCreate(&streams[i]);
	// }

	const int num_clusters = centroids->num_data_points;
	int N = points->num_data_points;

	// Find closest centroids for each datapoint
	const int num_threads_find_closest = 1024;
	const int num_blocks_find_closest = std::max(1, (int)std::ceil(points->num_data_points / num_threads_find_closest));
	const size_t shm_find_closest = sizeof(MyDataType) * num_clusters * N_FEATURES + sizeof(MyDataType) * num_threads_find_closest * N_FEATURES;
	if (MEASURE_TIME)
	{
		timer_find_closest_centroids->Start();
	}
	FindClosestCentroids<N_FEATURES><<<num_blocks_find_closest, num_threads_find_closest, shm_find_closest>>>(points->features_array,
																											  points->cluster_id_of_point, centroids->features_array, points->num_data_points,
																											  N_FEATURES, num_clusters);
	if (SYNCHRONIZE_AFTER_KERNEL_RUN)
	{
		cudaDeviceSynchronize();
	}
	if (MEASURE_TIME)
	{
		timer_find_closest_centroids->Stop();
		timer_find_closest_centroids->Elapsed();
	}
	cudaCheckError();
	// Find closest centroids for each datapoint

	// Create and init reduced points, what will be used sum up all points
	int num_threads = DEFAULT_NUM_THREADS;

	if (!RUN_REDUCE_FEATURE_WISE)
	{
		while (MAX_SHM_SIZE < CALCULATE_SHM_SIZE_JOIN_REDUCE(N_FEATURES, num_clusters, num_threads))
		{
			num_threads /= 2;
		}
	}
	else
	{
		while (MAX_SHM_SIZE < CALCULATE_SHM_SIZE_FEATURE_WISE_REDUCE(num_clusters, num_threads))
		{
			num_threads /= 2;
		}
		while (MAX_SHM_SIZE < CALCULATE_SHM_SIZE_COUNT(num_clusters, num_threads))
		{
			num_threads /= 2;
		}
	}

	int num_blocks = (int)std::max(std::ceil((long)(N / (double)num_threads / 2)), 1.0);

	const long num_reduced_points = num_blocks * num_clusters;

	if (cur_epoch == 0)
	{
		int num_threads_inti_id = (int)std::min(DEFAULT_NUM_THREADS, num_reduced_points);
		int num_block_init_id = (int)std::max(std::ceil((num_reduced_points / (double)num_threads_inti_id)), 1.0);
		reduced_points = AllocateDataPoints(N_FEATURES, num_reduced_points);
		InitPointsWithCentroidsIds<<<num_block_init_id, num_threads_inti_id>>>(reduced_points, num_clusters, num_reduced_points);
		cudaCheckError();
		cudaMallocManaged(&ids_count, sizeof(CountType) * num_reduced_points);
		cudaCheckError();
		cudaMemset(ids_count, 0, sizeof(CountType) * num_reduced_points);
		cudaCheckError();
	}
	else
	{
		cudaMemset(ids_count, 0, sizeof(CountType) * num_reduced_points);
		cudaCheckError();
	}
	// Create and init reduced points

	// reduce points in `points` and store them in reduced_poitsn
	if (MEASURE_TIME)
	{
		timer_compute_centroids->Start();
	}
	ReduceFeature<N_FEATURES>(points, reduced_points, ids_count, num_clusters, N, 1, &num_threads, &num_blocks, num_threads);
	if (SYNCHRONIZE_AFTER_KERNEL_RUN)
	{
		cudaDeviceSynchronize();
	}
	if (MEASURE_TIME)
	{
		timer_compute_centroids->Stop();
		timer_compute_centroids->Elapsed();
	}
	cudaCheckError();
	// reduce points in `points` and store them in reduced_poitsn
	if (DEBUG_GPU_ITERATION)
	{
		debugFunction<N_FEATURES>(points, ids_count, num_clusters, num_blocks, num_threads, num_clusters * num_blocks, "BEFORE WHILE REDUCE");
	}

	// further reduce points in `reduced_points`, until there will be no more then  `num_threads * 2` poitns left to reduce
	while (num_blocks * num_clusters > num_threads * 2)
	{
		N = num_blocks * num_clusters;
		num_blocks = GetNumBlocks(num_threads, num_blocks, num_clusters);

		if (MEASURE_TIME)
		{
			timer_compute_centroids->Start();
		}
		ReduceFeature<N_FEATURES>(reduced_points, reduced_points, ids_count, num_clusters, N, 0, &num_threads, &num_blocks, num_threads);
		if (SYNCHRONIZE_AFTER_KERNEL_RUN)
		{
			cudaDeviceSynchronize();
		}
		if (MEASURE_TIME)
		{
			timer_compute_centroids->Stop();
			timer_compute_centroids->Elapsed();
		}
		cudaCheckError();
	}
	if (DEBUG_GPU_ITERATION)
	{
		debugFunction<N_FEATURES>(points, ids_count, num_clusters, num_blocks, num_threads, num_blocks * num_clusters, "AFTER WHILE REDUCE");
	}
	// further reduce points in `reduced_points`, until there will be no more then  `num_threads * 2` poitns left to reduce

	// last reduce, reduce all remaining points to a 'single datapoint', that is: points belonging to the same cluster will be reduced to single point
	if (num_blocks > 1) // if may happen, that last reduced reduced all the point, happens when num_blocks==1
	{
		N = num_clusters * num_blocks;
		num_blocks = GetNumBlocks(num_threads, num_blocks, num_clusters);
		if (MEASURE_TIME)
		{
			timer_compute_centroids->Start();
		}
		ReduceFeature<N_FEATURES>(reduced_points, reduced_points, ids_count, num_clusters, N, 0, &num_threads, &num_blocks, std::ceil(N / 2.0));
		if (SYNCHRONIZE_AFTER_KERNEL_RUN)
		{
			cudaDeviceSynchronize();
		}

		if (MEASURE_TIME)
		{
			timer_compute_centroids->Stop();
			timer_compute_centroids->Elapsed();
		}
		cudaCheckError();
		if (DEBUG_GPU_ITERATION)
		{
			debugFunction<N_FEATURES>(points, ids_count, num_clusters, 1, num_threads, 1 * num_clusters, "AFTER LAST REDUCE");
		}
	}
	// last reduce, reduce all remaining points

	// find new centroids
	dim3 grid(1, 1, 1);
	dim3 block(N_FEATURES, num_clusters);
	FindNewCentroids<<<grid, block>>>(centroids, ids_count, reduced_points);
	cudaCheckError();
	// find new centroids

	// clean up, if it is last epoch
	if (cur_epoch + 1 == NUM_EPOCHES)
	{
		DeallocateDataPoints(reduced_points);
		cudaCheckError();
		cudaFree(ids_count);
		cudaCheckError();
		cur_epoch = 0;
	}
	else
	{
		cur_epoch++;
	}
	// clean up, if it is last epoch

	// for (int i = 0; i < NUM_STREAM; i++)
	// {
	// 	cudaStreamDestroy(streams[i]);
	// }
	return MeanSquareErrorParallel<N_FEATURES>(points, centroids);
}

template MyDataType KMeansOneIterationGpu<NUM_FEATURES>(DataPoints *points, DataPoints *centroids);