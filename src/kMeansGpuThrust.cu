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
	int N = points->num_data_points;
	int num_threads = 1024;
	int num_blocks = (int)std::max(std::ceil(((double)N / (double)num_threads)), 1.0);
	// get nearest clusters
	FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);
	// get nearest clusters
	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();

	// count number of points belonging to each cluster
	int count[centroids->num_data_points];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		count[c] = thrust::count(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, c);
		cudaCheckError();
	}
	// count number of points belonging to each cluster

	// find new positions of the clusters
	for (int feature = 0; feature < points->num_features; ++feature)
	{
		double *sumed_position_out; 
		cudaMallocManaged(&sumed_position_out, sizeof(double) * centroids->num_data_points);
		int *keys_out;
		cudaMallocManaged(&keys_out, sizeof(int) * centroids->num_data_points);

		int *keys_copy;
		cudaMallocManaged(&keys_copy, sizeof(int) * points->num_data_points);

		double *features_copy ;
		cudaMallocManaged(&features_copy, sizeof(double) * points->num_data_points);

		for (int i = 0; i < points->num_data_points; ++i)
		{
			keys_copy[i] = points->cluster_id_of_point[i];
			features_copy[i] = points->features_array[feature][i];
		}

		thrust::sort_by_key(keys_copy, keys_copy + points->num_data_points, features_copy);

		auto new_end = thrust::reduce_by_key(keys_copy, keys_copy + points->num_data_points, features_copy, keys_out, sumed_position_out);

		for (auto c = 0; c < centroids->num_data_points; c++)
		{
			centroids->features_array[feature][c] = sumed_position_out[c] / count[c];
		}

		cudaFree(keys_copy);
		cudaFree(features_copy);
		cudaFree(sumed_position_out);
		cudaFree(keys_out);
	}
	// find new positions of the clusters
}