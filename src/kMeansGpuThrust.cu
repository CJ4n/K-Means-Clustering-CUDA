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
	// init

	// int *nPoints = (int *)malloc(sizeof(int) * centroids->num_data_points);
	// double **sum = (double **)malloc(sizeof(double *) * centroids->num_features);

	// for (int feature = 0; feature < points->num_features; ++feature)
	// {
	// 	sum[feature] = (double *)malloc(sizeof(double) * centroids->num_data_points);
	// }
	// for (int c = 0; c < centroids->num_data_points; ++c)
	// {
	// 	nPoints[c] = 0;
	// 	std::vector<double> tmp;

	// 	for (int feature = 0; feature < points->num_features; ++feature)
	// 	{
	// 		sum[feature][c] = 0;
	// 	}
	// }

	// init

	// get nearest cluster
	int N = points->num_data_points;
	int num_threads = 1024;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	// size_t shmem_size = num_threads * sizeof(float);

	FindClosestCentroids<<<num_blocks, num_threads>>>(points, centroids);

	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();
		for (int i = 170; i < 200; i++)


	{
		std::cout << points->cluster_id_of_point[i] << ", ";
	}
	// thrust::device_vector<int> centroid_id_datapoint(points->num_data_points);
	// thrust::copy(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, centroid_id_datapoint.begin());
	// cudaCheckError();
	int count[centroids->num_data_points];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		// count[c] = thrust::count(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), c);
		count[c] = thrust::count(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, c);

		cudaCheckError();
	}


	for (int feature = 0; feature < points->num_features; ++feature)
	{
		double *sumed_position_out = (double *)malloc(sizeof(double) * centroids->num_data_points);
		memset(sumed_position_out, 0, centroids->num_data_points);
		int *keys_out = (int *)malloc(sizeof(int) * centroids->num_data_points);

		int *keys_copy = (int *)malloc(sizeof(int) * points->num_data_points);
		double *features_copy = (double *)malloc(sizeof(double) * points->num_data_points);

		for (int i = 0; i < points->num_data_points; ++i)
		{
			keys_copy[i] = points->cluster_id_of_point[i];

			features_copy[i] = points->features_array[feature][i];
		}

		thrust::sort_by_key(keys_copy, keys_copy + points->num_data_points, features_copy);

		auto new_end = thrust::reduce_by_key(keys_copy, keys_copy + points->num_data_points, features_copy, keys_out, sumed_position_out);

		// std::cout<<"feature: "<<feature<<" |";
		for (auto c = 0; c < centroids->num_data_points; c++)
		{
			centroids->features_array[feature][c] = sumed_position_out[c] / count[c];
			// std::cout<<centroids->features_array[feature][c]<<", ";
		}
		// std::cout<<std::endl;

		free(keys_copy);
		free(features_copy);
		free(sumed_position_out);
		free(keys_out);
	}
	// std::cout << std::endl;

	// std::cout << "-------------------------\n";
	// for (int i = 0; i < 200; i++)
	// {
	// 	std::cout << points->cluster_id_of_point[i] << ", ";
	// }
	// std::cout << std::endl;
	// for (int i = 0; i < centroids->num_data_points; i++)
	// {
	// 	std::cout << count[i] << ", ";
	// }
}