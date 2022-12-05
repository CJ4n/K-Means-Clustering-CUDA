#include "KMeansGpuThrust.h"

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include "CudaCheckError.h"

#include "FindClosestCentriods.h"
#include "Config.h"
#include "vector"
#include "string"

template <int F_NUM>
MyDataType KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids)
{
	const int N = points->num_data_points;
	const int num_threads = 1024;
	const int num_points = points->num_data_points;
	const int num_features = F_NUM;
	const int num_clusters = centroids->num_data_points;
	int num_blocks = (int)std::max(std::ceil(((double)N / (double)num_threads)), 1.0);
	// get nearest clusters
	const size_t shm_find_closest = sizeof(MyDataType) * num_clusters * NUM_FEATURES + sizeof(MyDataType) * num_threads * NUM_FEATURES;

	FindClosestCentroids<NUM_FEATURES><<<num_blocks, num_threads, shm_find_closest>>>(points->features_array, points->cluster_id_of_point, centroids->features_array, num_points, num_features, num_clusters);

	// get nearest clusters
	cudaDeviceSynchronize();
	cudaCheckError();

	// count number of points belonging to each cluster
	int count[centroids->num_data_points];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		count[c] = thrust::count(points->cluster_id_of_point, points->cluster_id_of_point + num_points, c);
		cudaCheckError();
	}
	// count number of points belonging to each cluster

	// find new positions of the clusters

	int **keys_copy;
	cudaMallocManaged(&keys_copy, sizeof(int *) * F_NUM);

	MyDataType **features_copy;
	cudaMallocManaged(&features_copy, sizeof(MyDataType *) * F_NUM);

	MyDataType **sumed_position_out;
	cudaMallocManaged(&sumed_position_out, sizeof(MyDataType *) * F_NUM);

	int **keys_out;
	cudaMallocManaged(&keys_out, sizeof(MyDataType *) * F_NUM);

	for (int f = 0; f < F_NUM; ++f)
	{
		cudaMallocManaged(&(keys_copy[f]), sizeof(int) * num_points);
		cudaCheckError();
		cudaMemcpyAsync(keys_copy[f], points->cluster_id_of_point, sizeof(int) * num_points, cudaMemcpyDeviceToDevice);
		cudaCheckError();

		cudaMallocManaged(&(features_copy[f]), sizeof(MyDataType) * num_points);
		cudaCheckError();
		cudaMemcpyAsync(features_copy[f], points->features_array[f], sizeof(MyDataType) * num_points, cudaMemcpyDeviceToDevice);
		cudaCheckError();

		cudaMallocManaged(&(sumed_position_out[f]), sizeof(MyDataType) * num_clusters);
		cudaCheckError();

		cudaMallocManaged(&(keys_out[f]), sizeof(int) * num_clusters);
		cudaCheckError();
	}

	cudaDeviceSynchronize();
	for (int f = 0; f < num_features; ++f)
	{
		// mozna by jakich prefetch zrobic tych danych tj. wczeniej poleciec async copy i miec odrazy wszyskie dane
		// co jeszce o tym pomyslec
		cudaCheckError();

		thrust::sort_by_key(keys_copy[f], keys_copy[f] + num_points, features_copy[f]);

		auto new_end = thrust::reduce_by_key(keys_copy[f], keys_copy[f] + num_points, features_copy[f], keys_out[f], sumed_position_out[f]);
		cudaCheckError();

		// if (feature + 1 < num_features)
		// {
		// 	cudaMemcpyAsync(keys_copy, points->cluster_id_of_point, sizeof(int) * num_points, cudaMemcpyDeviceToDevice);
		// 	cudaCheckError();

		// 	cudaMemcpyAsync(features_copy, points->features_array[feature + 1], sizeof(MyDataType) * num_points, cudaMemcpyDeviceToDevice);
		// 	cudaCheckError();
		// }
	}
	cudaDeviceSynchronize();
	for (int f = 0; f < num_features; ++f)
	{

		for (auto c = 0; c < num_clusters; c++)
		{
			centroids->features_array[f][c] = sumed_position_out[f][c] / count[c];
		}
	}
	cudaFree(keys_copy);
	cudaCheckError();
	cudaFree(features_copy);
	cudaCheckError();
	cudaFree(sumed_position_out);
	cudaCheckError();
	cudaFree(keys_out);
	cudaCheckError();
	return MeanSquareErrorParallel<F_NUM>(points, centroids);
	// find new positions of the clusters
}

template MyDataType KMeansOneIterationGpuThurst<NUM_FEATURES>(DataPoints *points, DataPoints *centroids);