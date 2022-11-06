#include "kMeansGpuThrust.h"

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include "cudaCheckError.h"

#include "findClosestCentriods.h"

#include "vector"
#include "string"


void k_means_one_iteration_gpu_thurst(dataPoints *points, dataPoints *centroids)
{
	// init

	int *nPoints = (int *)malloc(sizeof(int) * centroids->num_data_points);
	double **sum = (double **)malloc(sizeof(double *) * centroids->num_features);

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		sum[feature] = (double *)malloc(sizeof(double) * centroids->num_data_points);
	}
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		nPoints[c] = 0;
		std::vector<double> tmp;

		for (int feature = 0; feature < points->num_features; ++feature)
		{
			sum[feature][c] = 0;
		}
	}

	// init

	// get nearest cluster
	int N = points->num_data_points;
	int num_threads = 1024;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	// size_t shmem_size = num_threads * sizeof(float);

	find_closest_centroids<<<num_blocks, num_threads>>>(points, centroids);
	cudaDeviceSynchronize();

	cudaCheckError();

	thrust::device_vector<int> centroid_id_datapoint(points->num_data_points);
	thrust::copy(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, centroid_id_datapoint.begin());
	cudaCheckError();
	int count[5];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		count[c] = thrust::count(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), c);

		cudaCheckError();
	}
	std::cout << std::endl;

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		// thrust::device_vector<double> features(points->num_data_points);
		// thrust::device_vector<double> sum_position_of_centroid_featers_x(centroids->num_data_points);
		double *sumed_position = (double *)malloc(sizeof(double) * centroids->num_data_points);
		memset(sumed_position, 0, centroids->num_data_points);
		int *keys = (int *)malloc(sizeof(int) * centroids->num_data_points);
		// thrust::copy(points->features_array[feature], points->features_array[feature] + points->num_data_points-1, features.begin());
		// thrust::copy(points->features_array[feature], points->features_array[feature] + points->num_data_points, features.begin());
		thrust::sort_by_key(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, points->features_array[feature]);

		// auto val = features[points->num_data_points-1];
		// cudaCheckError();
		// thrust::reduce_by_key(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), features.begin(), sum_position_of_centroid_featers_x.begin(), sum_position_of_centroid_featers_x.end());
		auto new_end = thrust::reduce_by_key(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, points->features_array[feature], keys, sumed_position);
		// std::cout<<"val:"<<val<<std::endl;
		// cudaCheckError();
		// 	for(int p =0;p<200;p++){
		// 		std::cout<<"val: "<<points->features_array[feature][p]<<", id: "<<points->cluster_id_of_point[p]<<std::endl;
		// 	}
		// 	std::cout<<std::endl;

		// std::cout<<"{ ";
		for (auto c = 0; c < centroids->num_data_points; c++)
		{
			// std::cout<<*c<<std::endl;
			centroids->features_array[feature][c] = sumed_position[c] / count[c];
		}
		// 		std::cout<<" }"<<std::endl;
	}
}