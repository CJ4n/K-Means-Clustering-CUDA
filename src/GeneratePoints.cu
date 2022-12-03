#include "GeneratePoints.h"
#include <ctime>

#define MIN_FEATURE_VALUE 0
#define MAX_FEATURE_VALUE 35
DataPoints *GeneratePoints(const int num_features, const int num_points)
{
    DataPoints *data = AllocateDataPoints(num_features, num_points);
    data->num_features = num_features;
    data->num_data_points = num_points;

    for (int p = 0; p < num_points; ++p)
    {
        data->cluster_id_of_point[p] = 0;
        for (int f = 0; f < num_features; ++f)
        {
            const int range = MAX_FEATURE_VALUE - MIN_FEATURE_VALUE + 1;
            const int num = rand() % range + MIN_FEATURE_VALUE;
            data->features_array[f][p] = num;
        }
    }
    return data;
}