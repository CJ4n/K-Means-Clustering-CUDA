#pragma once
#include <string>

// struct Point
// {
// 	float x, y;	   // coordinates
// 	int cluster;   // no default cluster
// 	float minDist; // default infinite dist to nearest cluster

// 	Point() : x(0.0),
// 			  y(0.0),
// 			  cluster(-1),
// 			  minDist(__FLT_MAX__) {}

// 	Point(float x, float y) : x(x),
// 							  y(y),
// 							  cluster(-1),
// 							  minDist(__FLT_MAX__) {}
// };

typedef double MyDataType;
typedef int CountType;
struct DataPoints
{
	int *cluster_id_of_point;
	// int num_data_points;
	// int num_features;
	MyDataType **features_array;
};

DataPoints *AllocateDataPoints(const int num_features, const int num_data_points, const bool malloc_device = false);
void DeallocateDataPoints(DataPoints *data_points, int num_features);
MyDataType Distance(const DataPoints *p1, const DataPoints *p2, const int point_id, const int cluster_id, const int num_features);
MyDataType MeanSquareError(const DataPoints *point, const DataPoints *centroid, const int num_data_points, const int num_features);
// DataPoints *ReadCsv();
// void SaveCsv(const DataPoints *point, const std::string file_name);