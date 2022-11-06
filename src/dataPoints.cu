

#include <cuda.h>
#include <fstream>	// for file-reading
#include <iostream> // for file-reading
#include <sstream>	// for file-reading
#include <vector>
#include <dataPoints.h>
#include "cudaCheckError.h"


dataPoints *allocate_pt(int num_features, int num_data_points)
{

	dataPoints *point;
	cudaMallocManaged(&point, sizeof(dataPoints));
	cudaCheckError();

	point->num_data_points = num_data_points;
	cudaMallocManaged(&(point->cluster_id_of_point), sizeof(int) * num_data_points);
	cudaCheckError();
	cudaMallocManaged(&(point->minDist_to_cluster), sizeof(double) * num_data_points);
	cudaCheckError();

	point->num_features = num_features;
	cudaMallocManaged(&(point->features_array), sizeof(double *) * point->num_features);
	cudaCheckError();

	for (int feature = 0; feature < point->num_features; ++feature)
	{
		cudaMallocManaged(&(point->features_array[feature]), sizeof(double) * point->num_data_points);
		cudaCheckError();
	}
	return point;
}

double distance(dataPoints *p1, dataPoints *p2, int point_id, int cluster_id)
{
	double error = 0;
	for (int feature = 0; feature < p2->num_features; ++feature)
	{
		error += (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]) * (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]);
	}
	return error;
}

double MeanSquareError(dataPoints *point, dataPoints *centroid)
{
	double error = 0;
	for (int i = 0; i < point->num_data_points; ++i)
	{
		error += distance(centroid, point, i, point->cluster_id_of_point[i]);
	}
	return error / point->num_data_points;
}

dataPoints *readCsv()
{
	std::vector<Point> points;
	std::string line;
	std::ifstream file("/home/jan/Desktop/K-Means-Clustering-CUDA/mall_data.csv");
	// std::ifstream file("../mall_data.csv");
	while (std::getline(file, line))
	{
		std::stringstream lineStream(line);
		std::string bit;
		double x, y;
		getline(lineStream, bit, ',');
		x = std::stof(bit);
		getline(lineStream, bit, '\n');
		y = stof(bit);

		points.push_back(Point(x, y));
	}
	file.close();

	dataPoints *point = allocate_pt(2, points.size());
	int i = 0;
	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it)
	{
		double XY[2];
		XY[0] = it->x;
		XY[1] = it->y;
		for (int feature = 0; feature < point->num_features; ++feature)
		{
			point->features_array[feature][i] = XY[feature];
		}
		point->cluster_id_of_point[i] = it->cluster;
		point->minDist_to_cluster[i] = __DBL_MAX__;
		i++;
	}
	return point;
}

void saveCsv(dataPoints *point, std::string file_name)
{
	std::ofstream myfile;
	std::remove(file_name.c_str());
	myfile.open(file_name);
	myfile << "x,y,c" << std::endl;

	for (int i = 0; i < point->num_data_points; ++i)
	{
		for (int feature = 0; feature < point->num_features; ++feature)
		{
			myfile << point->features_array[feature][i];
			myfile << ",";
		}

		myfile << point->cluster_id_of_point[i] << std::endl;
	}
	myfile.close();
}