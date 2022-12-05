#pragma once

#include "dataPoints.h"

template <int N_FEATURES>
MyDataType KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids);
