#pragma once
#include <string>

struct Point
{
	float x, y;	// coordinates
	int cluster;	// no default cluster
	float minDist; // default infinite dist to nearest cluster

	Point() : x(0.0),
			  y(0.0),
			  cluster(-1),
			  minDist(__FLT_MAX__) {}

	Point(float x, float y) : x(x),
								y(y),
								cluster(-1),
								minDist(__FLT_MAX__) {}

	// double distance(Point p)
	// {
	// 	return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
	// }
};

struct DataPoints
{
	float **features_array;
	int *cluster_id_of_point;
	float *minDist_to_cluster;
	int num_data_points;
	int num_features;
};

DataPoints *AllocateDataPoints(int num_features, int num_data_points);
void DeallocateDataPoints(DataPoints* data_points);
float Distance(DataPoints *p1, DataPoints *p2, int point_id, int cluster_id);
float MeanSquareError(DataPoints *point, DataPoints *centroid);
DataPoints *ReadCsv();
void SaveCsv(DataPoints *point, std::string file_name);