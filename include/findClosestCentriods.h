#pragma once
#include <cuda.h>
#include "dataPoints.h"
template <int F_NUM>
__global__ void FindClosestCentroids(MyDataType **features,int *centroids_ids,  MyDataType** centeriods_features,const int num_points,const int num_features,const int num_clusters);