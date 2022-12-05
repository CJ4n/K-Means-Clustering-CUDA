#pragma once
#include <cuda.h>

#include "dataPoints.h"

MyDataType KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids);
