// #pragma once
#include <cuda.h>
#include "dataPoints.h"

void KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids, const int num_clusters, const int num_data_points);
