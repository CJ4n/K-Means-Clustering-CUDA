
#include "findClosestCentriods.h"

__global__ void FindClosestCentroids(DataPoints *points, DataPoints *centroids)
{
	// int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	float min_dist = __FLT_MAX__;
	if (gid >= points->num_data_points)
	{
		return;
	}
	
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		if (points->num_data_points < gid)
		{
			return;
		}
		float dist = 0;
		for (int feature = 0; feature < centroids->num_features; ++feature)
		{
			float tmp = points->features_array[feature][gid] - centroids->features_array[feature][c];
			dist += tmp * tmp;
		}

		if (dist < min_dist)
		{
			min_dist = dist;
			points->cluster_id_of_point[gid] = c;
		}
	}
}