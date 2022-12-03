#pragma once
#include "dataPoints.h"
template <int N_FEATURES>
void KMeansOneIterationCpu(DataPoints *points, DataPoints *centroids, const int num_clusters, const int num_data_points);