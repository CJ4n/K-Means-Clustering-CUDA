// #pragma once
#include <cuda.h>
#include "dataPoints.h"

void k_means_one_iteration_gpu_thurst(dataPoints *points, dataPoints *centroids);
