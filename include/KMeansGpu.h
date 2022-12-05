#pragma once

#include "DataPoints.h"

template <int N_FEATURES>
MyDataType KMeansOneIterationGpu(DataPoints *points, DataPoints *centroids);
