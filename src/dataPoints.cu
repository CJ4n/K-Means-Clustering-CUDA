#pragma once

#include <cuda.h>
#include <fstream>	// for file-reading
#include <iostream> // for file-reading
#include <sstream>	// for file-reading
#include <vector>
#include <dataPoints.h>
#include "cudaCheckError.h"

DataPoints *AllocateDataPoints(int num_features, int num_data_points)
{

	DataPoints *point;
	cudaMallocManaged(&point, sizeof(DataPoints));
	cudaCheckError();

	point->num_data_points = num_data_points;
	cudaMallocManaged(&(point->cluster_id_of_point), sizeof(int) * num_data_points);
	cudaCheckError();
	// for(int i = 0;i<num_data_points;i++){
	// 	point->cluster_id_of_point[i]=1;
	// }
	cudaMallocManaged(&(point->minDist_to_cluster), sizeof(float) * num_data_points);
	cudaCheckError();

	point->num_features = num_features;
	cudaMallocManaged(&(point->features_array), sizeof(float *) * point->num_features);
	cudaCheckError();

	for (int feature = 0; feature < point->num_features; ++feature)
	{
		cudaMallocManaged(&(point->features_array[feature]), sizeof(*(point->features_array[feature])) * point->num_data_points);
		cudaCheckError();
	}
	return point;
}

void DeallocateDataPoints(DataPoints *data_points )
{
	for (int f = 0; f < data_points->num_features; f++)
	{
		cudaFree(data_points->features_array[f]);
	}
	cudaFree(data_points->features_array);
	cudaFree(data_points->cluster_id_of_point);
	cudaFree(data_points->minDist_to_cluster);
	cudaFree(data_points);
}

float Distance(DataPoints *p1, DataPoints *p2, int point_id, int cluster_id)
{
	float error = 0;
	for (int feature = 0; feature < p2->num_features; ++feature)
	{
		error += (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]) * (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]);
	}
	return error;
}

float MeanSquareError(DataPoints *point, DataPoints *centroid)
{
	float error = 0;
	for (int i = 0; i < point->num_data_points; ++i)
	{
		error += Distance(centroid, point, i, point->cluster_id_of_point[i]);
	}
	return error / point->num_data_points;
}

DataPoints *ReadCsv()
{
	std::vector<Point> points;
	std::string line;
	std::ifstream file("/home/jan/Desktop/K-Means-Clustering-CUDA/mall_data.csv");
	// std::ifstream file("../mall_data.csv");
	while (std::getline(file, line))
	{
		std::stringstream lineStream(line);
		std::string bit;
		float x, y;
		getline(lineStream, bit, ',');
		x = std::stof(bit);
		getline(lineStream, bit, '\n');
		y = stof(bit);

		points.push_back(Point(x, y));
	}
	file.close();

	DataPoints *point = AllocateDataPoints(2, points.size());
	int i = 0;
	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it)
	{
		float XY[2];
		XY[0] = it->x;
		XY[1] = it->y;
		for (int feature = 0; feature < point->num_features; ++feature)
		{
			point->features_array[feature][i] = XY[feature];
		}
		point->cluster_id_of_point[i] = it->cluster;
		point->minDist_to_cluster[i] = __FLT_MAX__;
		i++;
	}
	return point;
}

void SaveCsv(DataPoints *point, std::string file_name)
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