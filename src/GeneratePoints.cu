#include "GeneratePoints.h"

#include <Timer.h>

#include "Config.h"

DataPoints *GeneratePoints(const int num_features, const int num_points)
{
    timer_data_generation->total_time = 0;
    timer_data_generation->Start();

    DataPoints *data = AllocateDataPoints(num_features, num_points);
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
    timer_data_generation->Stop();
    timer_data_generation->Elapsed();
    return data;
}
