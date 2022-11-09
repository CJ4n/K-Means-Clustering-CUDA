#pragma once
#include <cuda.h>
#include "dataPoints.h"

__global__ void FindClosestCentroids(DataPoints *points, DataPoints *centroids);