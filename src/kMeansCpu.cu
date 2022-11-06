#include "kMeansCpu.h"
// #pragma once

#include "vector"
#include "string"

void k_means_one_iteration_cpu(dataPoints *points, dataPoints *centroids)
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

	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			double dist = distance(centroids, points, p, c);
			if (dist < points->minDist_to_cluster[p])
			{
				points->minDist_to_cluster[p] = dist;
				points->cluster_id_of_point[p] = c;
			}
		}
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
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		for (int feature = 0; feature < points->num_features; ++feature)
		{
			centroids->features_array[feature][c] = sum[feature][c] / nPoints[c];
		}
	}
	// get centroids new location

	// find new clusters
}