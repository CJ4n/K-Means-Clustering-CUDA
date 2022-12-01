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
								 const int count_in, CountType *count_out, const int num_data_points, const int num_features, const int num_clusters)
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

#define INDEX_ID_TEST(c, tid, num_clusters) (c) + tid *num_clusters

__global__ void ReduceDataPointsTEST(MyDataType **features, int *cluster_ids, MyDataType **centroids_features,
									 const int count_in, CountType *count_out, const int num_data_points, const int num_features, const int num_clusters)
{
	extern __shared__ CountType shm_1[];
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

	for (int c = 0; c < num_clusters; ++c)
	{
		shm_1[INDEX_ID_TEST(c, tid, num_clusters)] = 0;
	}

	c1 = cluster_ids[gid];

	if (gid + blockDim.x < num_data_points)
	{
		c2 = cluster_ids[gid + blockDim.x];
	}
	if (count_in)
		shm_1[INDEX_ID_TEST(c1, tid, num_clusters)] = count_in;
	else
		shm_1[INDEX_ID_TEST(c1, tid, num_clusters)] = count_out[gid];

	if (gid + blockDim.x < num_data_points)
	{
		if (count_in)
			shm_1[INDEX_ID_TEST(c2, tid, num_clusters)] += count_in;
		else
			shm_1[INDEX_ID_TEST(c2, tid, num_clusters)] += count_out[gid + blockDim.x];
	}
	// else
	// 	return;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (tid < stride)
		{
			for (int c = 0; c < num_clusters; ++c)
			{
				shm_1[INDEX_ID_TEST(c, tid, num_clusters)] += shm_1[INDEX_ID_TEST(c, (tid + stride), num_clusters)];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			// [{count1,...,count5},{count1,...,count5},..,
			count_out[blockIdx.x * num_clusters + c] = shm_1[INDEX_ID_TEST(c, 0, num_clusters)];
		}
	}
}

#define INDEX_C(c, tid, num_clusters) c + (tid * num_clusters)
__global__ void ReduceDataPointsCountPoints(const int *cluster_ids,
											const CountType count_in, CountType *count_out, const int num_data_points, const int num_clusters)
{
	extern __shared__ CountType shm_c[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (gid >= num_data_points)
	{
		return;
	}

	int c1, c2;

	for (int c = 0; c < num_clusters; ++c)
	{
		shm_c[INDEX_C(c, tid, num_clusters)] = 0;
	}

	c1 = cluster_ids[gid];
	if (gid + blockDim.x < num_data_points)
	{
		c2 = cluster_ids[gid + blockDim.x];
	}

	if (count_in)
		shm_c[INDEX_C(c1, tid, num_clusters)] = count_in;
	else
		shm_c[INDEX_C(c1, tid, num_clusters)] = count_out[gid];

	if (gid + blockDim.x < num_data_points)
	{
		if (count_in)
			shm_c[INDEX_C(c2, tid, num_clusters)] += count_in;
		else
			shm_c[INDEX_C(c2, tid, num_clusters)] += count_out[gid + blockDim.x];
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
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

	if (tid == 0)
	{
		for (int c = 0; c < num_clusters; ++c)
		{
			count_out[blockIdx.x * num_clusters + c] = shm_c[INDEX_C(c, 0, num_clusters)];
		}
	}
}

#define INDEX_F(c, tid, num_clusters) c + (tid * num_clusters)

__global__ void ReduceDataPointsByFeatures(MyDataType *features, int *cluster_ids, MyDataType *out,
										   const int num_data_points, const int num_clusters)
{
	extern __shared__ MyDataType shm_f[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (gid >= num_data_points)
	{
		return;
	}

	int c1, c2;

	for (int c = 0; c < num_clusters; ++c)
	{
		shm_f[INDEX_F(c, tid, num_clusters)] = 0;
	}

	c1 = cluster_ids[gid];
	if (gid + blockDim.x < num_data_points)
	{
		c2 = cluster_ids[gid + blockDim.x];
	}

	shm_f[INDEX_F(c1, tid, num_clusters)] = features[gid];
	if (gid + blockDim.x < num_data_points)
	{
		shm_f[INDEX_F(c2, tid, num_clusters)] += features[gid + blockDim.x];
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
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

int RoundToPowerOf2(int n)
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
}

#define DEBUG 1
#define MAX_SHM_SIZE 48 * 1024
#define DEFAULT_NUM_THREADS 1024l
#define CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads) num_threads *(num_features + 1) * num_clusters * sizeof(MyDataType)

DataPoints *reduced_points;
int cur_epoch = 0;

void debugFunction(DataPoints *points, CountType *ids_count, int num_features, int num_clusters, int num_blocks, int num_threads, int N, std::string label)
{

	std::cout << "\n---------" << label << "---------" << std::endl;
	DataPoints *debug = AllocateDataPoints(num_features, num_clusters);

	long sum_tot = 0;
	// Gets exact sum by feature and clusters
	for (int f = 0; f < num_features; f++)
		for (int c = 0; c < num_clusters; c++)
		{
			debug->features_array[f][c] = 0;
		}

	for (int i = 0; i < points->num_data_points; i++)
	{
		for (int f = 0; f < num_features; f++)
		{
			debug->features_array[f][points->cluster_id_of_point[i]] += points->features_array[f][i];
			sum_tot += points->features_array[f][i];
		}
	}

	std::cout << " correct (points)\n{\n	";
	double sum_tot_v2 = 0;
	for (int c = 0; c < num_clusters; c++)
	{
		for (int f = 0; f < num_features; f++)
		{
			std::cout << debug->features_array[f][c] << ", ";
			sum_tot_v2 += debug->features_array[f][c];
			debug->features_array[f][c] = 0;
		}
	}
	std::cout << "\n}\n";
	// Gets exact sum by feature and clusters

	// Gets redcued sum by feature and cluster
	long sum_tot_reduced = 0;

	for (int i = 0; i < num_blocks * num_clusters; i++)
	{
		for (int f = 0; f < num_features; f++)
		{
			debug->features_array[f][reduced_points->cluster_id_of_point[i]] += reduced_points->features_array[f][i];
			sum_tot_reduced += reduced_points->features_array[f][i];
		}
	}
	std::cout << "Calculated points (reduced_points)\n{\n	";

	double sum_tot_reduced_v2 = 0;
	for (int c = 0; c < num_clusters; c++)
	{
		for (int f = 0; f < num_features; f++)
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

// zmiana liczy features wpływa na wynik xD
void ReduceFeature(DataPoints *points, CountType *ids_count, int num_features, int num_clusters,
				   int N, CountType count_in, int *num_th, int *num_bl)
{
	int num_threads = *num_th;
	int num_blocks = *num_bl; //(int)std::ceil(N / (float)num_threads / 2.0);
	// num_blocks = RoundToPowerOf2(num_blocks);
	// *num_th = num_threads;
	// *num_bl = num_blocks;

	size_t shm_size = sizeof(MyDataType) * num_threads * num_clusters;
	for (int f = 0; f < num_features; ++f)
	{
		cudaDeviceSynchronize();

		ReduceDataPointsByFeatures<<<num_blocks, num_threads, shm_size>>>(points->features_array[f],
																		  points->cluster_id_of_point, reduced_points->features_array[f],
																		  N, num_clusters);
		cudaDeviceSynchronize();
		cudaCheckError();
	}
	cudaDeviceSynchronize();
	shm_size = sizeof(CountType) * num_threads * num_clusters;
	ReduceDataPointsCountPoints<<<num_blocks, num_threads, shm_size>>>(points->cluster_id_of_point,
																	   count_in, ids_count, N, num_clusters);
	cudaDeviceSynchronize();
	cudaCheckError();
	// if (DEBUG)
	// 	debugFunction(points, ids_count, num_features, num_clusters, num_blocks, num_threads, num_blocks * num_clusters,
	// 				  "INSE______________________");
	
		cudaCheckError();
}

void KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids)
{
	int num_features = points->num_features;
	const int num_clusters = centroids->num_data_points;
	int N = points->num_data_points;
	// num_features=1;

	// Find closest centroids for each datapoint
	const int num_threads_find_closest = 1024;
	const int num_blocks_find_closest = std::max(1, (int)std::ceil(points->num_data_points / num_threads_find_closest));
	const size_t shm_find_closest = sizeof(MyDataType) * num_clusters * num_features + sizeof(MyDataType) * num_threads_find_closest * num_features;
	timer_find_closest_centroids->Start();
	// FindClosestCentroids<<<num_blocks_find_closest, num_threads_find_closest, shm_find_closest>>>(points->features_array,
	// 																							  points->cluster_id_of_point, centroids->features_array, points->num_data_points,
	// 																							  num_features, num_clusters);
	timer_find_closest_centroids->Stop();
	timer_find_closest_centroids->Elapsed();
	cudaCheckError();
	cudaDeviceSynchronize();

	for (int p = 0; p < points->num_data_points; ++p)
	{
		MyDataType min_dist = __DBL_MAX__;
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			MyDataType dist = Distance(centroids, points, p, c);
			if (dist < min_dist)
			{
				min_dist = dist;
				points->cluster_id_of_point[p] = c;
			}
		}
		int cid = points->cluster_id_of_point[p];
	}

	// Find closest centroids for each datapoint

	// Create and init reduced points, what will be used sum up all points
	int num_threads = DEFAULT_NUM_THREADS;
	// while (MAX_SHM_SIZE < CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads))
	// {
	// 	num_threads /= 2;
	// }

	size_t shmem_size = CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads);
	
	int num_blocks = (int)std::max(std::ceil((long)(N / (double)num_threads / 2)), 1.0);
	// int num_blocks = (int)std::max(std::ceil((long)(N / (double)1024 / 2)), 1.0);
	CountType *ids_count;

	// tmp = lambda(tmp);
	long num_reduced_points = num_blocks * num_clusters;
	// if (cur_epoch == 0)
	{
		reduced_points = AllocateDataPoints(num_features, num_reduced_points);
		int num_threads_inti_id = (int)std::min(DEFAULT_NUM_THREADS, num_reduced_points);
		int num_block_init_id = (int)std::max(std::ceil((num_reduced_points / (double)num_threads_inti_id)), 1.0);
		InitPointsWithCentroidsIds<<<num_block_init_id, num_threads_inti_id>>>(reduced_points, num_clusters, num_reduced_points);
		cudaDeviceSynchronize();
		cudaCheckError();
		cudaMallocManaged(&ids_count, sizeof(CountType) * num_blocks * num_clusters);
		cudaCheckError();
		cudaMemset(ids_count, 0, sizeof(CountType) * num_blocks * num_clusters);
		cudaCheckError();
	}
	// else
	{
		// cudaMemset(ids_count, 0, sizeof(CountType) * num_blocks * num_clusters);

		// clean recued_points??
	}
	// Create and init reduced points

	// if (DEBUG)
	// {
	// 	debugFunction(points, num_features, num_clusters, num_blocks, num_threads, num_clusters * num_blocks, "BEFORE FIRST REDUCE");
	// }
	// reduce points in `points` and store them in `reduced_poitsn`
	timer_compute_centroids->Start();
	// ReduceDataPoints<<<num_blocks, num_threads, shmem_size>>>(points->features_array,
	// 														  points->cluster_id_of_point, reduced_points->features_array,
	// 														  1, ids_count, N, num_features, num_clusters);

	ReduceFeature(points, ids_count, num_features, num_clusters, N, 1, &num_threads, &num_blocks);

	timer_compute_centroids->Stop();
	timer_compute_centroids->Elapsed();
	cudaCheckError();
	// reduce points in `points` and store them in reduced_poitsn

	if (DEBUG)
	{
		debugFunction(points, ids_count, num_features, num_clusters, num_blocks, num_threads, num_clusters * num_blocks, "BEFORE WHILE REDUCE");
	}
	// further reduce points in `reduced_points`, until there will be no more then  `num_threads * 2` poitns left to reduce
	while (num_blocks * num_clusters > num_threads * 2)
	{
		 N = num_blocks * num_clusters;
		// N=lambda(N);
		num_blocks =GetNumBlocks(num_threads,num_blocks,num_clusters); //std::ceil(N / num_threads / 2.0);
		//shmem_size =  sizeof(MyDataType) * num_threads * num_clusters;;//CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads);

		// timer_compute_centroids->Start();
		// ReduceDataPoints<<<num_blocks, num_threads, shmem_size>>>(reduced_points->features_array,
		// 														  reduced_points->cluster_id_of_point, reduced_points->features_array,
		// 														  0, ids_count, N, num_features, num_clusters);
		ReduceFeature(reduced_points, ids_count, num_features, num_clusters, N, 0, &num_threads, &num_blocks);
		// timer_compute_centroids->Stop();
		// timer_compute_centroids->Elapsed();
cudaDeviceSynchronize();
		cudaCheckError();
	}
	if (DEBUG)
	{
		debugFunction(points, ids_count, num_features, num_clusters, num_blocks, num_threads, num_blocks * num_clusters, "AFTER WHILE REDUCE");
	}
	// further reduce points in `reduced_points`, until there will be no more then  `num_threads * 2` poitns left to reduce

	// last reduce, reduce all remaining points to a 'single datapoint', that is: points belonging to the same cluster will be reduced to single point
	// if (num_blocks > 1) // if may happen, that last reduced reduced all the point, happen when num_blocks==1
	// {
	// 	N = num_clusters * num_blocks;
	// 	int num_threads_last_sumup = std::ceil(N / 2.0);
	// 	num_threads_last_sumup = RoundToPowerOf2(num_threads_last_sumup);

	// 	shmem_size = CALCULATE_SHM_SIZE(num_features, num_clusters, num_threads_last_sumup);
	// 	timer_compute_centroids->Start();
	// 	// ReduceDataPoints<<<1, num_threads_last_sumup, shmem_size>>>(reduced_points->features_array,
	// 	// 															reduced_points->cluster_id_of_point, reduced_points->features_array,
	// 	// 															0, ids_count, N, num_features, num_clusters);
	// 	timer_compute_centroids->Stop();
	// 	timer_compute_centroids->Elapsed();
	// 	cudaCheckError();
	// 	if (DEBUG)
	// 	{
	// 		debugFunction(points, ids_count, num_features, num_clusters, 1, num_threads_last_sumup, 1 * num_clusters, "AFTER LAST REDUCE");
	// 	}
	// }
	// last reduce, reduce all remaining points

	// find new centroids
	FindNewCentroids<<<1, num_features * num_clusters>>>(centroids, ids_count, reduced_points);
	cudaDeviceSynchronize();
	cudaCheckError();
	// find new centroids

	// cleanup memory
	// if (cur_epoch++ == constants::num_epoches - 1)
	{
		DeallocateDataPoints(reduced_points);
		cudaCheckError();

		cudaFree(ids_count);
		cudaCheckError();
		cur_epoch = 0;
	}

	cudaCheckError();
}