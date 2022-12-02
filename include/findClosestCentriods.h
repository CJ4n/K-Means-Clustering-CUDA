#pragma once
#include <cuda.h>
#include "dataPoints.h"
__global__ void FindClosestCentroids(MyDataType **features,int *centroids_ids,  MyDataType** centeriods_features,const int num_points,const int num_features,const int num_clusters);