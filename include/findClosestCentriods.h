#pragma once
#include <cuda.h>
#include "dataPoints.h"

__global__ void find_closest_centroids(dataPoints *points, dataPoints *centroids);