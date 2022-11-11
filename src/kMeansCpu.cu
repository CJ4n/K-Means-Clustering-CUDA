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
	float **sum = (float **)malloc(sizeof(float *) * centroids->num_features);

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		sum[feature] = (float *)malloc(sizeof(float) * centroids->num_data_points);
	}
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		nPoints[c] = 0;
		std::vector<float> tmp;

		for (int feature = 0; feature < points->num_features; ++feature)
		{
			sum[feature][c] = 0;
		}
	}
	// init

	// get nearest cluster
	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			float dist = Distance(centroids, points, p, c);
			float min_dist = points->minDist_to_cluster[p];
			if (dist < min_dist)
			{
				points->minDist_to_cluster[p] = dist;
				points->cluster_id_of_point[p] = c;
			}
		}
		int cid = points->cluster_id_of_point[p];

		points->minDist_to_cluster[p] = __DBL_MAX__;
	}
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
	// sum all points 'belonging' to each centroid

	// get centroids new location
	for (int feature = 0; feature < points->num_features; ++feature)
	{
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			centroids->features_array[feature][c] = sum[feature][c] / nPoints[c];
		}
	}
	// get centroids new location

	free(nPoints);

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		free(sum[feature]);
	}
	free(sum);
}