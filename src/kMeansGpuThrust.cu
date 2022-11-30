#include "kMeansGpuThrust.h"

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include "cudaCheckError.h"

#include "findClosestCentriods.h"

#include "vector"
#include "string"

void KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids)
{
	const int N = points->num_data_points;
	const int num_threads = 1024;
	const int num_points = points->num_data_points;
	const int num_features=points->num_features;
	const int num_clusters= centroids->num_data_points;
	int num_blocks = (int)std::max(std::ceil(((double)N / (double)num_threads)), 1.0);
	// get nearest clusters
	const size_t shm_find_closest = sizeof(MyDataType) * num_clusters*num_features;


	// FindClosestCentroids<<<num_blocks, num_threads,shm_find_closest>>>(points->features_array,points->cluster_id_of_point, centroids->features_array,num_points,num_features,num_clusters);

	
	// get nearest clusters
	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();

	// count number of points belonging to each cluster
	int count[centroids->num_data_points];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		count[c] = thrust::count(points->cluster_id_of_point, points->cluster_id_of_point +num_points, c);
		cudaCheckError();
	}
	// count number of points belonging to each cluster

	// find new positions of the clusters

	int *keys_copy;
	cudaMallocManaged(&keys_copy, sizeof(int) *num_points);
	cudaMemcpyAsync(keys_copy, points->cluster_id_of_point, sizeof(int) *num_points, cudaMemcpyDeviceToDevice);
	cudaCheckError();

	float *features_copy;
	cudaMallocManaged(&features_copy, sizeof(float) * num_points);
	cudaMemcpyAsync(features_copy, points->features_array[0], sizeof(float) * num_points, cudaMemcpyDeviceToDevice);
	cudaCheckError();

	float *sumed_position_out;
	cudaMallocManaged(&sumed_position_out, sizeof(float) * num_clusters);
	int *keys_out;
	cudaMallocManaged(&keys_out, sizeof(int) * num_clusters);

	for (int feature = 0; feature < num_features; ++feature)
	{
		// mozna by jakich prefetch zrobic tych danych tj. wczeniej poleciec async copy i miec odrazy wszyskie dane
		// co jeszce o tym pomyslec
		cudaDeviceSynchronize();

		thrust::sort_by_key(keys_copy, keys_copy + num_points, features_copy);

		auto new_end = thrust::reduce_by_key(keys_copy, keys_copy + num_points, features_copy, keys_out, sumed_position_out);


		if (feature + 1 <num_features)
		{
			cudaMemcpyAsync(keys_copy, points->cluster_id_of_point, sizeof(int) * num_points, cudaMemcpyDeviceToDevice);
			cudaCheckError();

			cudaMemcpyAsync(features_copy, points->features_array[feature + 1], sizeof(float) * num_points, cudaMemcpyDeviceToDevice);
			cudaCheckError();
		}

		for (auto c = 0; c < num_clusters; c++)
		{
			centroids->features_array[feature][c] = sumed_position_out[c] / count[c];
		}
	}
	cudaFree(keys_copy);
	cudaFree(features_copy);
	cudaFree(sumed_position_out);
	cudaFree(keys_out);
	// find new positions of the clusters
}