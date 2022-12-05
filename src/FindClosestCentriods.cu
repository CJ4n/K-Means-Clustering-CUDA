#include "FindClosestCentriods.h"

#include "Config.h"

#define INDEX_CLUSTER(f, c, num_clusters) ((f * num_clusters) + c)
#define INDEX_POINT(f, tid, num_clusters, num_features) (num_features * num_clusters + tid * num_features + f)
template <int F_NUM>
__global__ void FindClosestCentroids(MyDataType **features, int *centroids_ids, MyDataType **centeriods_features, const int num_points, const int num_features, const int num_clusters)
{
	//  centroids				| data points
	// (f1{c1,c2,c3}f2{c1,c2,c3}|f1{c1,c2,c3}f2{c1,c2,c3},...,f1{c1,c2,c3}f2{c1,c2,c3})
	extern __shared__ MyDataType shm2[];
	const int tid = threadIdx.x;
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid >= num_points)
	{
		return;
	}

	if (tid < num_clusters * F_NUM)
	{
		int x = tid / num_clusters;
		int y = tid % num_clusters;
		shm2[INDEX_CLUSTER(x, y, num_clusters)] = centeriods_features[x][y];
	}

	for (int f = 0; f < F_NUM; ++f)
	{
		shm2[INDEX_POINT(f, tid, num_clusters, F_NUM)] = features[f][gid];
	}
	MyDataType min_dist = __DBL_MAX__;

	__syncthreads();
	int cur_centroids = -1;

	for (int c = 0; c < num_clusters; ++c)
	{
		MyDataType dist = 0;
		for (int f = 0; f < F_NUM; ++f)
		{
			MyDataType tmp = shm2[INDEX_POINT(f, tid, num_clusters, F_NUM)] - shm2[INDEX_CLUSTER(f, c, num_clusters)];
			dist += tmp * tmp;
		}

		if (dist < min_dist)
		{
			min_dist = dist;
			cur_centroids = c;
		}
	}

	centroids_ids[gid] = cur_centroids;
}

template __global__ void FindClosestCentroids<NUM_FEATURES>(MyDataType **features, int *centroids_ids, MyDataType **centeriods_features, const int num_points, const int num_features, const int num_clusters);
