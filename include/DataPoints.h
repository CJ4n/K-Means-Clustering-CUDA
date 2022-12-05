#pragma once

typedef double MyDataType;
typedef int CountType;
struct DataPoints
{
	int *cluster_id_of_point;
	int num_data_points;
	// int num_features;
	MyDataType **features_array;
};

DataPoints *AllocateDataPoints(const int num_features, const int num_data_points);
void DeallocateDataPoints(DataPoints *data_points, const int num_features);
MyDataType Distance(const DataPoints *p1, const DataPoints *p2, const int point_id, const int cluster_id, const int num_features);
MyDataType MeanSquareError(const DataPoints *point, const DataPoints *centroid, const int num_features);
template <int F_NUM>
MyDataType MeanSquareErrorParallel(const DataPoints *point, const DataPoints *centroid);