#include "GeneratePoints.h"
#include <ctime>

#define MIN_FEATURE_VALUE 0
#define MAX_FEATURE_VALUE 35
DataPoints *GeneratePoints(int num_features, int num_points)
{
    DataPoints *data = AllocateDataPoints(num_features, num_points);
    data->num_features=num_features;
    data->num_data_points=num_points;
    for (int p = 0; p < num_points; ++p)
    {
        data->cluster_id_of_point[p] = -1;
        data->minDist_to_cluster[p] = __DBL_MAX__;
        for (int f = 0; f < num_features; ++f)
        {
            int range = MAX_FEATURE_VALUE - MIN_FEATURE_VALUE + 1;
            int num = rand() % range + MIN_FEATURE_VALUE;
            data->features_array[f][p] = num;
        }
    }
    return data;
}