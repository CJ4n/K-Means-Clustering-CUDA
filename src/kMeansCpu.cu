#include "kMeansCpu.h"
// #pragma once

#include <vector>
#include <string>
#include <iostream>
#include "findClosestCentriods.h"
#include "Constants.h"
template <int N_FEATURES>
void KMeansOneIterationCpu(DataPoints *points, DataPoints *centroids, const int num_clusters, const int num_data_points)
{
	// init
	int *nPoints = (int *)malloc(sizeof(int) * num_clusters);
	double **sum = (double **)malloc(sizeof(*(points->features_array)) * N_FEATURES);

	for (int feature = 0; feature <N_FEATURES; ++feature)
	{
		sum[feature] = (double *)malloc(sizeof(**(points->features_array)) * num_clusters);
	}
	for (int c = 0; c < num_clusters; ++c)
	{
		nPoints[c] = 0;
		std::vector<double> tmp;

		for (int feature = 0; feature < N_FEATURES; ++feature)
		{
			sum[feature][c] = 0;
		}
	}
	// init

	// get nearest cluster
	for (int p = 0; p < num_data_points; ++p)
	{
		MyDataType min_dist= __DBL_MAX__;
		for (int c = 0; c < num_clusters; ++c)
		{
			MyDataType dist = Distance(centroids, points, p, c,N_FEATURES);
			if (dist < min_dist)
			{
				min_dist = dist;
				points->cluster_id_of_point[p] = c;
			}
		}
		int cid = points->cluster_id_of_point[p];

	}
	// get nearest cluster

	// sum all points 'belonging' to each centroid
	for (int p = 0; p < num_data_points; ++p)
	{
		for (int feature = 0; feature <N_FEATURES; ++feature)
		{
			sum[feature][points->cluster_id_of_point[p]] += points->features_array[feature][p];
		}
		nPoints[points->cluster_id_of_point[p]]++;
	}
	// sum all points 'belonging' to each centroid

	// get centroids new location
	for (int feature = 0; feature < N_FEATURES; ++feature)
	{
		for (int c = 0; c <num_clusters; ++c)
		{
			centroids->features_array[feature][c] = sum[feature][c] / (double)nPoints[c];
		}
	}
	// get centroids new location

	free(nPoints);

	for (int feature = 0; feature < N_FEATURES; ++feature)
	{
		free(sum[feature]);
	}
	free(sum);
}

template void KMeansOneIterationCpu<NUM_FEATURES>(DataPoints *points, DataPoints *centroids,const int num_clusters,const int num_data_points);