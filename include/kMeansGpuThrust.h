// #pragma once
#include <cuda.h>
#include "dataPoints.h"

void KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids);