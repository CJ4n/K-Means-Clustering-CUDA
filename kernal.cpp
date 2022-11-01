#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <ctime>	// for a random seed
#include <fstream>	// for file-reading
#include <iostream> // for file-reading
#include <sstream>	// for file-reading
#include <vector>

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
	double **features;
	int *cluster;
	double *minDist;
	int n;
	int features_number;
};

double distance(Point p1, Point p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
pt readCsv()
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

	pt point;
	int n = points.size();
	point.n = n;
	point.cluster = (int *)malloc(sizeof(int) * n);
	point.minDist = (double *)malloc(sizeof(double) * n);
	point.features_number = 2;
	point.features = (double **)malloc(sizeof(double *) * point.features_number);
	for (int feature = 0; feature < point.features_number; ++feature)
	{
		point.features[feature] = (double *)malloc(sizeof(double) * point.n);
	}
	int i = 0;
	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it)
	{
		double XY[2];
		XY[0] = it->x;
		XY[1] = it->y;
		for (int feature = 0; feature < point.features_number; ++feature)
		{
			point.features[feature][i] = XY[feature];
		}
		point.cluster[i] = it->cluster;
		point.minDist[i] = __DBL_MAX__;
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

	for (int i = 0; i < point->n; ++i)
	{
		for (int feature = 0; feature < point->features_number; ++feature)
		{
			myfile << point->features[feature][i];
			myfile << ",";
		}

		myfile << point->cluster[i] << std::endl;
	}
	myfile.close();
}

double distance(pt *p1, pt *p2, int point_id, int cluster_id)
{
	double error = 0;
	for (int feature = 0; feature < p2->features_number; ++feature)
	{
		error += (p1->features[feature][cluster_id] - p2->features[feature][point_id]) * (p1->features[feature][cluster_id] - p2->features[feature][point_id]);
	}
	return error;
}
double MeanSquareError(pt *point, pt *centroid)
{
	double error = 0;
	for (int i = 0; i < point->n; ++i)
	{
		error += distance(centroid, point, i, point->cluster[i]);
	}
	return error / point->n;
}
void k_means_iteration(pt *points, pt *centroids, int num_clusters)
{
	std::vector<int> nPoints;
	std::vector<std::vector<double>> sum;
	for (int j = 0; j < num_clusters; ++j)
	{
		nPoints.push_back(0);
		std::vector<double> tmp;
		for (int feature = 0; feature < points->features_number; ++feature)
		{
			tmp.push_back(0.0);
		}
		sum.push_back(tmp);
	}

	for (int p = 0; p < points->n; ++p)
	{
		for (int c = 0; c < centroids->n; ++c)
		{
			double dist = distance(centroids, points, p, c);
			if (dist < points->minDist[p])
			{
				points->minDist[p] = dist;
				points->cluster[p] = c;
			}
		}
		for (int feature = 0; feature < points->features_number; ++feature)
		{
			sum[points->cluster[p]][feature] += points->features[feature][p];
		}
		nPoints[points->cluster[p]]++;
	}
	for (int c = 0; c < centroids->n; ++c)
	{
		for (int feature = 0; feature < points->features_number; ++feature)
		{
			centroids->features[feature][c] = sum[c][feature] / nPoints[c];
		}
	}
}

void kMeansClustering(pt *point, int epochs, int num_clusters)
{
	pt centroids;
	centroids.n = point->n;
	centroids.features_number = point->features_number;
	centroids.features = (double **)malloc(sizeof(double *) * centroids.features_number);
	for (int feature = 0; feature < centroids.features_number; ++feature)
	{
		centroids.features[feature] = (double *)malloc(sizeof(double) * centroids.n);
	}

	centroids.cluster = (int *)malloc(num_clusters * sizeof(double));
	std::srand(time(0)); // need to set the random seed
	for (int i = 0; i < num_clusters; ++i)
	{
		int n = rand() % point->n;
		for (int feature = 0; feature < point->features_number; ++feature)
		{
			centroids.features[feature][i] = point->features[feature][n];
		}

		centroids.cluster[i] = i;
	}
	centroids.n = num_clusters;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		k_means_iteration(point, &centroids, num_clusters);
		std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, &centroids) << std::endl;
		saveCsv(point, "train" + std::to_string(epoch) + ".csv");
	}
}
int main(int argc, char **argv)
{
	auto point = readCsv();
	kMeansClustering(&point, 6, 5);
	saveCsv(&point, "output.csv");

	return 0;
}
