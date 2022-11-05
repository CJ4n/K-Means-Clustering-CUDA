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

#define cudaCheckError()                                                                    \
	{                                                                                       \
		cudaError_t e = cudaGetLastError();                                                 \
		if (e != cudaSuccess)                                                               \
		{                                                                                   \
			printf("Cudafailure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0);                                                                        \
		}                                                                                   \
	}

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

struct pt
{
	double **features_array;
	int *cluster_id_of_point;
	double *minDist_to_cluster;
	int num_data_points;
	int features_number;
};

pt *readCsv()
{
	std::vector<Point> points;
	std::string line;
	std::ifstream file("mall_data.csv");
	std::getline(file, line);
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

	pt *point;
	cudaMallocManaged(&point, sizeof(pt));
	int n = points.size();
	point->num_data_points = n;
	// point->cluster_id_of_point = (int *)malloc(sizeof(int) * n);
	cudaMallocManaged(&(point->cluster_id_of_point), sizeof(int) * n);
	cudaCheckError();
	// point->minDist_to_cluster = (double *)malloc(sizeof(double) * n);
	cudaMallocManaged(&(point->minDist_to_cluster), sizeof(double) * n);
	cudaCheckError();

	point->features_number = 2;
	// point->features_array = (double **)malloc(sizeof(double *) * point->features_number);
	cudaMallocManaged(&(point->features_array), sizeof(double *) * point->features_number);
	cudaCheckError();

	for (int feature = 0; feature < point->features_number; ++feature)
	{
		// point->features_array[feature] = (double *)malloc(sizeof(double) * point->num_data_points);
		cudaMallocManaged(&(point->features_array[feature]), sizeof(double) * point->num_data_points);
		cudaCheckError();
	}
	int i = 0;
	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it)
	{
		double XY[2];
		XY[0] = it->x;
		XY[1] = it->y;
		for (int feature = 0; feature < point->features_number; ++feature)
		{
			point->features_array[feature][i] = XY[feature];
		}
		point->cluster_id_of_point[i] = it->cluster;
		point->minDist_to_cluster[i] = __DBL_MAX__;
		i++;
	}
	return point;
}

void saveCsv(pt *point, std::string file_name)
{
	std::ofstream myfile;
	std::remove(file_name.c_str());
	myfile.open(file_name);
	myfile << "x,y,c" << std::endl;

	for (int i = 0; i < point->num_data_points; ++i)
	{
		for (int feature = 0; feature < point->features_number; ++feature)
		{
			myfile << point->features_array[feature][i];
			myfile << ",";
		}

		myfile << point->cluster_id_of_point[i] << std::endl;
	}
	myfile.close();
}

double distance(pt *p1, pt *p2, int point_id, int cluster_id)
{
	double error = 0;
	for (int feature = 0; feature < p2->features_number; ++feature)
	{
		error += (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]) * (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]);
	}
	return error;
}

double MeanSquareError(pt *point, pt *centroid)
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

__global__ void find_closest_centroids(pt *points, pt *centroids)
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
		for (int feature = 0; feature < centroids->features_number; ++feature)
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

void k_means_one_iteration(pt *points, pt *centroids)
{
	// init

	int *nPoints = (int *)malloc(sizeof(int) * centroids->num_data_points);
	double **sum = (double **)malloc(sizeof(double *) * centroids->features_number);

	for (int feature = 0; feature < points->features_number; ++feature)
	{
		sum[feature] = (double *)malloc(sizeof(double) * centroids->num_data_points);
	}
	for (int c = 0; c < centroids->num_data_points; ++c)
	{
		nPoints[c] = 0;
		std::vector<double> tmp;

		for (int feature = 0; feature < points->features_number; ++feature)
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

	// for (int p = 0; p < points->num_data_points; ++p)
	// {
	// 	for (int c = 0; c < centroids->num_data_points; ++c)
	// 	{
	// 		double dist = distance(centroids, points, p, c);
	// 		if (dist < points->minDist_to_cluster[p])
	// 		{
	// 			points->minDist_to_cluster[p] = dist;
	// 			points->cluster_id_of_point[p] = c;
	// 		}
	// 	}
	// }

	// get nearest cluster

	// find new clusters

	// sum all points 'belonging' to each centroid
	for (int p = 0; p < points->num_data_points; ++p)
	{
		for (int feature = 0; feature < points->features_number; ++feature)
		{
			sum[feature][points->cluster_id_of_point[p]] += points->features_array[feature][p];
		}
		nPoints[points->cluster_id_of_point[p]]++;
	}

	thrust::device_vector<int> centroid_id_datapoint(points->num_data_points);
	thrust::copy(points->cluster_id_of_point, points->cluster_id_of_point + points->num_data_points, centroid_id_datapoint.begin());
	cudaCheckError();
	int count[5];
	for (int c = 0; c < centroids->num_data_points; c++)
	{
		count[c] = thrust::count(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), c);
	
		cudaCheckError();
	}
	std::cout<<std::endl;

	for (int feature = 0; feature < points->features_number; ++feature)
	{
		thrust::device_vector<double> features(points->num_data_points);
		thrust::device_vector<double> sum_position_of_centroid_featers_x(centroids->num_data_points);

		thrust::copy(points->features_array[feature], points->features_array[feature] + points->num_data_points, features.begin());
		cudaCheckError();
		thrust::reduce_by_key(centroid_id_datapoint.begin(), centroid_id_datapoint.end(), features.begin(), sum_position_of_centroid_featers_x.begin(), sum_position_of_centroid_featers_x.end());
		cudaCheckError();

		for (int c = 0; c < centroids->num_data_points; c++)
		{
			centroids->features_array[feature][c] = sum_position_of_centroid_featers_x[c] / count[c];
			std::cout<<sum_position_of_centroid_featers_x[c]<<", ";
		}
		std::cout<<std::endl;
	}

	// sum all points 'belonging' to each centroid

	// get centroids new location
	// for (int c = 0; c < centroids->num_data_points; ++c)
	// {
	// 	for (int feature = 0; feature < points->features_number; ++feature)
	// 	{
	// 		centroids->features_array[feature][c] = sum[feature][c] / nPoints[c];
	// 	}
	// }
	// get centroids new location

	// find new clusters
}

void kMeansClustering(pt *point, int epochs, int num_clusters)
{
	pt *centroids;
	cudaMallocManaged(&centroids, sizeof(pt));
	cudaCheckError();

	centroids->num_data_points = point->num_data_points;
	centroids->features_number = point->features_number;
	// centroids.features_array = (double **)malloc(sizeof(double *) * centroids.features_number);

	cudaMallocManaged(&(centroids->features_array), sizeof(double *) * centroids->features_number);
	cudaCheckError();

	for (int feature = 0; feature < centroids->features_number; ++feature)
	{
		// centroids.features_array[feature] = (double *)malloc(sizeof(double) * centroids.num_data_points);
		cudaMallocManaged(&(centroids->features_array[feature]), sizeof(double) * centroids->num_data_points);
		cudaCheckError();
	}

	centroids->cluster_id_of_point = (int *)malloc(num_clusters * sizeof(double));
	cudaMallocManaged(&(centroids->cluster_id_of_point), num_clusters * sizeof(double));
	cudaCheckError();
	std::srand(time(0)); // need to set the random seed
	for (int i = 0; i < num_clusters; ++i)
	{
		int n = rand() % point->num_data_points;
		for (int feature = 0; feature < point->features_number; ++feature)
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
		saveCsv(point, "train" + std::to_string(epoch) + ".csv");
		k_means_one_iteration(point, centroids);
	}
}
int main(int argc, char **argv)
{
	pt *point = readCsv();
	kMeansClustering(point, 6, 5);
	saveCsv(point, "output.csv");

	return 0;
}
