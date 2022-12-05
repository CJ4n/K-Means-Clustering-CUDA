#pragma once
#include <cuda.h>

#include "DataPoints.h"

MyDataType KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids);
