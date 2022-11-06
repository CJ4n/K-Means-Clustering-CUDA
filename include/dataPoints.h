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

	double distance(Point p)
	{
		return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
	}
};

struct dataPoints
{
	double **features_array;
	int *cluster_id_of_point;
	double *minDist_to_cluster;
	int num_data_points;
	int num_features;
};

dataPoints *allocate_pt(int num_features, int num_data_points);
double distance(dataPoints *p1, dataPoints *p2, int point_id, int cluster_id);
double MeanSquareError(dataPoints *point, dataPoints *centroid);
dataPoints *readCsv();
void saveCsv(dataPoints *point, std::string file_name);