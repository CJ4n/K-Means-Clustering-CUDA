#pragma once
#include <string>

struct Point
{
	double x, y;	// coordinates
	int cluster;	// no default cluster
	double minDist; // default infinite dist to nearest cluster

	Point() : x(0.0),
			  y(0.0),
			  cluster(-1),
			  minDist(__DBL_MAX__) {}

	Point(double x, double y) : x(x),
								y(y),
								cluster(-1),
								minDist(__DBL_MAX__) {}

	// double distance(Point p)
	// {
	// 	return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
	// }
};

struct DataPoints
{
	double **features_array;
	int *cluster_id_of_point;
	double *minDist_to_cluster;
	int num_data_points;
	int num_features;
};

DataPoints *AllocateDataPoints(int num_features, int num_data_points);
void DeallocateDataPoints(DataPoints* data_points);
double Distance(DataPoints *p1, DataPoints *p2, int point_id, int cluster_id);
double MeanSquareError(DataPoints *point, DataPoints *centroid);
DataPoints *ReadCsv();
void SaveCsv(DataPoints *point, std::string file_name);