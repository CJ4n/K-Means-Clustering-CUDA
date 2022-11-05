

#include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <ctime>	// for a random seed
#include <fstream>	// for file-reading
#include <iostream> // for file-reading
#include <sstream>	// for file-reading
#include <vector>
#include <math.h>
// #include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "dataPoints.h"

#define cudaCheckError()                                                                    \
	{                                                                                       \
		cudaError_t e = cudaGetLastError();                                                 \
		if (e != cudaSuccess)                                                               \
		{                                                                                   \
			printf("Cudafailure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0);                                                                        \
		}                                                                                   \
	}

// struct Point
// {
// 	double x, y;	// coordinates
// 	int cluster;	// no default cluster
// 	double minDist; // default infinite dist to nearest cluster

// 	Point() : x(0.0),
// 			  y(0.0),
// 			  cluster(-1),
// 			  minDist(__DBL_MAX__) {}

// 	Point(double x, double y) : x(x),
// 								y(y),
// 								cluster(-1),
// 								minDist(__DBL_MAX__) {}

// 	double distance(Point p)
// 	{
// 		return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
// 	}
// };

// struct pt
// {
// 	double **features_array;
// 	int *cluster_id_of_point;
// 	double *minDist_to_cluster;
// 	int num_data_points;
// 	int num_features;
// };

// pt *allocate_pt(int num_features, int num_data_points)
// {

// 	pt *point;
// 	cudaMallocManaged(&point, sizeof(pt));
// 	cudaCheckError();

// 	point->num_data_points = num_data_points;
// 	cudaMallocManaged(&(point->cluster_id_of_point), sizeof(int) * num_data_points);
// 	cudaCheckError();
// 	cudaMallocManaged(&(point->minDist_to_cluster), sizeof(double) * num_data_points);
// 	cudaCheckError();

// 	point->num_features = num_features;
// 	cudaMallocManaged(&(point->features_array), sizeof(double *) * point->num_features);
// 	cudaCheckError();

// 	for (int feature = 0; feature < point->num_features; ++feature)
// 	{
// 		cudaMallocManaged(&(point->features_array[feature]), sizeof(double) * point->num_data_points);
// 		cudaCheckError();
// 	}
// 	return point;
// }

// pt *readCsv()
// {
// 	std::vector<Point> points;
// 	std::string line;
// 	std::ifstream file("/home/jan/Desktop/K-Means-Clustering-CUDA/mall_data.csv");
// 	// std::ifstream file("../mall_data.csv");
// 	while (std::getline(file, line))
// 	{
// 		std::stringstream lineStream(line);
// 		std::string bit;
// 		double x, y;
// 		getline(lineStream, bit, ',');
// 		x = std::stof(bit);
// 		getline(lineStream, bit, '\n');
// 		y = stof(bit);

// 		points.push_back(Point(x, y));
// 	}
// 	file.close();

// 	pt *point = allocate_pt(2, points.size());
// 	int i = 0;
// 	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it)
// 	{
// 		double XY[2];
// 		XY[0] = it->x;
// 		XY[1] = it->y;
// 		for (int feature = 0; feature < point->num_features; ++feature)
// 		{
// 			point->features_array[feature][i] = XY[feature];
// 		}
// 		point->cluster_id_of_point[i] = it->cluster;
// 		point->minDist_to_cluster[i] = __DBL_MAX__;
// 		i++;
// 	}
// 	return point;
// }

// void saveCsv(pt *point, std::string file_name)
// {
// 	std::ofstream myfile;
// 	std::remove(file_name.c_str());
// 	myfile.open(file_name);
// 	myfile << "x,y,c" << std::endl;

// 	for (int i = 0; i < point->num_data_points; ++i)
// 	{
// 		for (int feature = 0; feature < point->num_features; ++feature)
// 		{
// 			myfile << point->features_array[feature][i];
// 			myfile << ",";
// 		}

// 		myfile << point->cluster_id_of_point[i] << std::endl;
// 	}
// 	myfile.close();
// }

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

// __device__ void distance(double *features_point, double *features_centroid, int num_features, double *distance_out)
// {
// 	*distance_out = 0;
// 	for (int feature = 0; feature < num_features; ++feature)
// 	{
// 		double tmp = features_point[feature] - features_centroid[feature];
// 		*distance_out += tmp * tmp;
// 	}
// }

__global__ void find_closest_centroids(dataPoints *points, dataPoints *centroids)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		if (points->num_data_points < gid)
		{
			return;
		}
		int dist = 0;
		for (int feature = 0; feature < centroids->num_features; ++feature)
		{
			double tmp = points->features_array[feature][gid] - centroids->features_array[feature][c];
			dist += tmp * tmp;
		}

		if (dist < points->minDist_to_cluster[gid])
		{
			points->minDist_to_cluster[gid] = dist;
			points->cluster_id_of_point[gid] = c;
		}
	}
}

void k_means_one_iteration_gpu_thurst(dataPoints *points, dataPoints *centroids)
{
	// init

	int *nPoints = (int *)malloc(sizeof(int) * centroids->num_data_points);
	double **sum = (double **)malloc(sizeof(double *) * centroids->num_features);

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		sum[feature] = (double *)malloc(sizeof(double) * centroids->num_data_points);
	}
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		nPoints[c] = 0;
		std::vector<double> tmp;

		for (int feature = 0; feature < points->num_features; ++feature)
		{
			sum[feature][c] = 0;
		}
	}

	// init

	// get nearest cluster
	int N = points->num_data_points;
	int num_threads = 1024;
	int num_blocks = (int)std::max(std::ceil((int)(N / num_threads)), 1.0);
	// size_t shmem_size = num_threads * sizeof(float);

	find_closest_centroids<<<num_blocks, num_threads>>>(points, centroids);
	cudaDeviceSynchronize();

	cudaCheckError();

	thrust::device_vector<int> centroid_id_datapoint(points->num_data_points);
	thrust::copy(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, centroid_id_datapoint.begin());
	cudaCheckError();
	int count[5];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		count[c] = thrust::count(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), c);

		cudaCheckError();
	}
	std::cout << std::endl;

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		// thrust::device_vector<double> features(points->num_data_points);
		// thrust::device_vector<double> sum_position_of_centroid_featers_x(centroids->num_data_points);
		double *sumed_position = (double *)malloc(sizeof(double) * centroids->num_data_points);
		memset(sumed_position, 0, centroids->num_data_points);
		int *keys = (int *)malloc(sizeof(int) * centroids->num_data_points);
		// thrust::copy(points->features_array[feature], points->features_array[feature] + points->num_data_points-1, features.begin());
		// thrust::copy(points->features_array[feature], points->features_array[feature] + points->num_data_points, features.begin());
		thrust::sort_by_key(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, points->features_array[feature]);

		// auto val = features[points->num_data_points-1];
		// cudaCheckError();
		// thrust::reduce_by_key(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), features.begin(), sum_position_of_centroid_featers_x.begin(), sum_position_of_centroid_featers_x.end());
		auto new_end = thrust::reduce_by_key(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, points->features_array[feature], keys, sumed_position);
		// std::cout<<"val:"<<val<<std::endl;
		// cudaCheckError();
		// 	for(int p =0;p<200;p++){
		// 		std::cout<<"val: "<<points->features_array[feature][p]<<", id: "<<points->cluster_id_of_point[p]<<std::endl;
		// 	}
		// 	std::cout<<std::endl;

		// std::cout<<"{ ";
		for (auto c = 0; c < centroids->num_data_points; c++)
		{
			// std::cout<<*c<<std::endl;
			centroids->features_array[feature][c] = sumed_position[c] / count[c];
		}
		// 		std::cout<<" }"<<std::endl;
	}
}

// void kMeansClusteringGPUThrust(pt *point, int epochs, int num_clusters)
// {
// 	pt *centroids = allocate_pt(point->num_features, num_clusters);
// 	cudaCheckError();
// 	// std::srand(time(0)); // need to set the random seed
// 	for (int i = 0; i < num_clusters; ++i)
// 	{
// 		int n = rand() % point->num_data_points;
// 		for (int feature = 0; feature < point->num_features; ++feature)
// 		{
// 			centroids->features_array[feature][i] = point->features_array[feature][n];
// 		}

// 		centroids->cluster_id_of_point[i] = i;
// 	}
// 	centroids->num_data_points = num_clusters;

// 	// alloc cuda memory

// 	for (int epoch = 0; epoch < epochs; ++epoch)
// 	{
// 		std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, centroids) << std::endl;
// 		saveCsv(point, "train" + std::to_string(epoch) + ".csv");
// 		k_means_one_iteration_gpu_thurst(point, centroids);
// 	}
// }

void k_means_one_iteration_cpu(dataPoints *points, dataPoints *centroids)
{
	// init

	int *nPoints = (int *)malloc(sizeof(int) * centroids->num_data_points);
	double **sum = (double **)malloc(sizeof(double *) * centroids->num_features);

	for (int feature = 0; feature < points->num_features; ++feature)
	{
		sum[feature] = (double *)malloc(sizeof(double) * centroids->num_data_points);
	}
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		nPoints[c] = 0;
		std::vector<double> tmp;

		for (int feature = 0; feature < points->num_features; ++feature)
		{
			sum[feature][c] = 0;
		}
	}

	// init

	// get nearest cluster

	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int c = 0; c < centroids->num_data_points; ++c)
		{
			double dist = distance(centroids, points, p, c);
			if (dist < points->minDist_to_cluster[p])
			{
				points->minDist_to_cluster[p] = dist;
				points->cluster_id_of_point[p] = c;
			}
		}
	}

	// get nearest cluster

	// sum all points 'belonging' to each centroid
	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int feature = 0; feature < points->num_features; ++feature)
		{
			sum[feature][points->cluster_id_of_point[p]] += points->features_array[feature][p];
		}
		nPoints[points->cluster_id_of_point[p]]++;
	}

	// sum all points 'belonging' to each centroid

	// get centroids new location
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		for (int feature = 0; feature < points->num_features; ++feature)
		{
			centroids->features_array[feature][c] = sum[feature][c] / nPoints[c];
		}
	}
	// get centroids new location

	// find new clusters
}
void kMeansClustering(dataPoints *point, int epochs, int num_clusters, void (*k_means_one_iteration_algorithm)(dataPoints *, dataPoints *))
{
	dataPoints *centroids = allocate_pt(point->num_features, num_clusters);
	cudaCheckError();
	for (int i = 0; i < num_clusters; ++i)
	{
		int n = rand() % point->num_data_points;
		for (int feature = 0; feature < point->num_features; ++feature)
		{
			centroids->features_array[feature][i] = point->features_array[feature][n];
		}

		centroids->cluster_id_of_point[i] = i;
	}
	centroids->num_data_points = num_clusters;

	// alloc cuda memory

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, centroids) << std::endl;
		// saveCsv(point, "train" + std::to_string(epoch) + ".csv");
		k_means_one_iteration_algorithm(point, centroids);
	}
}
int main(int argc, char **argv)
{
	dataPoints *point = readCsv();
	std::srand(time(0)); // need to set the random seed

	kMeansClustering(point, 6, 5,k_means_one_iteration_gpu_thurst);
	cudaFree(point);
	std::cout << "----------------\n";
	point = readCsv();
	kMeansClustering(point, 6, 5, k_means_one_iteration_cpu);

	saveCsv(point, "output.csv");

	return 0;
}
