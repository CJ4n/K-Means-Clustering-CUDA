
#include "findClosestCentriods.h"

__global__ void FindClosestCentroids(DataPoints *points, DataPoints *centroids)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		if (points->num_data_points < gid)
		{
			return;
		}
		double dist = 0;
		for (int feature = 0; feature < centroids->num_features; ++feature)
		{
			double tmp = points->features_array[feature][gid] - centroids->features_array[feature][c];
			dist += tmp * tmp;
		}

		if (dist < points->minDist_to_cluster[gid])
		{
			points->minDist_to_cluster[gid] = dist;
			points->cluster_id_of_point[gid] = c;
		}
	}
	if (points->num_data_points > gid)
	{
		points->minDist_to_cluster[gid] = __DBL_MAX__;
	}
}