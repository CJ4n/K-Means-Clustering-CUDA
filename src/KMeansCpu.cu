#include "KMeansCpu.h"

#include "Config.h"

template <int F_NUM>
MyDataType KMeansOneIterationCpu(DataPoints *points, DataPoints *centroids)
{
	// init
	int *nPoints = (int *)malloc(sizeof(int) * centroids->num_data_points);
	double **sum = (double **)malloc(sizeof(*(points->features_array)) * F_NUM);

	for (int feature = 0; feature < F_NUM; ++feature)
	{
		sum[feature] = (double *)malloc(sizeof(**(points->features_array)) * centroids->num_data_points);
	}
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		nPoints[c] = 0;

		for (int feature = 0; feature < F_NUM; ++feature)
		{
			sum[feature][c] = 0;
		}
	}
	// init

	// get nearest cluster
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
	}
	// get nearest cluster

	// sum all points 'belonging' to each centroid
	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int feature = 0; feature < F_NUM; ++feature)
		{
			sum[feature][points->cluster_id_of_point[p]] += points->features_array[feature][p];
		}
		nPoints[points->cluster_id_of_point[p]]++;
	}
	// sum all points 'belonging' to each centroid

	// get centroids new location
	for (int feature = 0; feature < F_NUM; ++feature)
	{
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			centroids->features_array[feature][c] = sum[feature][c] / (double)nPoints[c];
		}
	}
	// get centroids new location

	free(nPoints);

	for (int feature = 0; feature < F_NUM; ++feature)
	{
		free(sum[feature]);
	}
	free(sum);

	return MeanSquareError(points, centroids);
}

template MyDataType KMeansOneIterationCpu<NUM_FEATURES>(DataPoints *points, DataPoints *centroids);