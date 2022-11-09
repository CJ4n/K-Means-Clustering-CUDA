#include "kMeansCpu.h"
// #pragma once

#include <vector>
#include <string>
#include <iostream>
#include "findClosestCentriods.h"

void KMeansOneIterationCpu(DataPoints *points, DataPoints *centroids)
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
	// for (int i = 0; i < 20; i++)
	// {
	// 	int cid = points->cluster_id_of_point[i];
	// 	int aaa = points->cluster_id_of_point[i];

	// 	std::cout << cid<<", ";

	// }
	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			double dist = Distance(centroids, points, p, c);
			double min_dist = points->minDist_to_cluster[p];
			if (dist < min_dist)
			{
				points->minDist_to_cluster[p] = dist;
				points->cluster_id_of_point[p] = c;
			}
		}
		int cid = points->cluster_id_of_point[p];

		points->minDist_to_cluster[p] = __DBL_MAX__;
	}
	for (int i = 170; i < 200; i++)
	{
				std::cout<<points->cluster_id_of_point[i]<<", ";
	}
	// for (int i = 0; i < 20; i++)
	// {
	// 	int cid = points->cluster_id_of_point[i];
	// 	int aaa = points->cluster_id_of_point[i];

	// 	std::cout << cid<<", ";
	// }
	// std::cout << "\n-----------------------------" << std::endl;

	// 	int N = points->num_data_points;
	// int num_threads = 1024;
	// int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	// // size_t shmem_size = num_threads * sizeof(float);

	// find_closest_centroids<<<num_blocks, num_threads>>>(points, centroids);
	// cudaDeviceSynchronize();

	// get nearest cluster

	// sum all points 'belonging' to each centroid
	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int feature = 0; feature < points->num_features; ++feature)
		{
			sum[feature][points->cluster_id_of_point[p]] += points->features_array[feature][p];
		}
		nPoints[points->cluster_id_of_point[p]]++;
	}
	// for (int i = 0; i < centroids->num_data_points; i++)
	// {
	// 	std::cout << nPoints[i] << ", ";
	// }
	// sum all points 'belonging' to each centroid
	// std::cout<<std::endl;
	// get centroids new location
	for (int feature = 0; feature < points->num_features; ++feature)
	{
		// std::cout<<"feature: "<<feature<<" |";

		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			centroids->features_array[feature][c] = sum[feature][c] / nPoints[c];
			// std::cout<<centroids->features_array[feature][c]<<", ";
		}
		// std::cout<<std::endl;
	}

	// get centroids new location

	// find new clusters
	// std::cout << "points count: ";
	// for (int i = 0; i < centroids->num_data_points; i++)
	// {
	// 	std::cout << nPoints[i] << ", ";
	// }
}