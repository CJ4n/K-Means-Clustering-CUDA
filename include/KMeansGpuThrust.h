#pragma once
#include <cuda.h>

#include "DataPoints.h"

template <int F_NUM>
MyDataType KMeansOneIterationGpuThurst(DataPoints *points, DataPoints *centroids);
