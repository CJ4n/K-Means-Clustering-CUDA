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
	double *x;
	double *y;
	int *cluster;
	double *minDist;
	int n;
};

double distance(Point p1, Point p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
pt readCsv()
{
	std::vector<Point> points;
	// pt point;
	// point.x = (double *)malloc(sizeof(double) * n);
	// point.y = (double *)malloc(sizeof(double) * n);
	// point.cluster = (int *)malloc(sizeof(int) * n);
	// point.minDist = (double *)malloc(sizeof(double) * n);

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
	point.x = (double *)malloc(sizeof(double) * n);
	point.y = (double *)malloc(sizeof(double) * n);
	point.cluster = (int *)malloc(sizeof(int) * n);
	point.minDist = (double *)malloc(sizeof(double) * n);
	int i = 0;
	for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it)
	{
		point.x[i] = it->x;
		point.y[i] = it->y;
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

	// for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
	for (int i = 0; i < point->n; ++i)
	{
		myfile << point->x[i] << "," << point->y[i] << "," << point->cluster[i] << std::endl;
	}
	myfile.close();
}

double distance(double x1, double y1, double x2, double y2)
{
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}
double MeanSquareError(pt *point, pt *centroid)
{
	double error = 0;
	// for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
	for (int i = 0; i < point->n; ++i)
	{
		error += distance(centroid->x[point->cluster[i]], centroid->y[point->cluster[i]], point->x[i], point->y[i]);
	}
	return error / point->n;
}
void findClosestCentroids(pt *points, pt *centroids, int k)
{
	std::vector<int> nPoints;
	std::vector<double> sumX, sumY;
	for (int j = 0; j < k; ++j)
	{
		nPoints.push_back(0);
		sumX.push_back(0.0);
		sumY.push_back(0.0);
	}

	// for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it)
	for (int p = 0; p < points->n; ++p)
	{
		// for (std::vector<Point>::iterator c = std::begin(*centroids); c != std::end(*centroids); ++c)
		for (int c = 0; c < centroids->n; ++c)
		{
			// quick hack to get cluster index
			// int clusterId = c - std::begin(*centroids);

			// Point p = *it;
			double dist = distance(centroids->x[c], centroids->y[c], points->x[p], points->y[p]);
			if (dist < points->minDist[p])
			{
				points->minDist[p] = dist;
				points->cluster[p] = c;
			}
			// *it = p;
		}
		sumX[points->cluster[p]] += points->x[p];
		sumY[points->cluster[p]] += points->y[p];
		nPoints[points->cluster[p]]++;
	}
	// for (std::vector<Point>::iterator c = std::begin(*centroids); c != std::end(*centroids); ++c)
	for (int c = 0; c < centroids->n; ++c)

	{
		// int clusterId = c - std::begin(*centroids);
		centroids->x[c] = sumX[c] / nPoints[c];
		centroids->y[c] = sumY[c] / nPoints[c];
		// std::cout << "cluster: " << c->x << ", " << c->y << std::endl;
	}
}

void kMeansClustering(pt *point, int epochs, int k)
{
	pt centroids;
	centroids.x = (double *)malloc(k * sizeof(double));
	centroids.y = (double *)malloc(k * sizeof(double));
	centroids.cluster = (int *)malloc(k * sizeof(double));
	std::srand(time(0)); // need to set the random seed
	for (int i = 0; i < k; ++i)
	{
		// centroids.push_back(points->at(rand() % points->size()));
		int n = rand() % point->n;
		centroids.x[i] = point->x[n];
		centroids.y[i] = point->y[n];
		centroids.cluster[i] = i;
	}
	centroids.n = k;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{

		findClosestCentroids(point, &centroids, k);
		std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, &centroids) << std::endl;

		// std::vector<int> nPoints;
		// std::vector<double> sumX, sumY;

		// // Initialise with zeroes
		// for (int j = 0; j < k; ++j)
		// {
		// 	nPoints.push_back(0);
		// 	sumX.push_back(0.0);
		// 	sumY.push_back(0.0);
		// }
		// Iterate over points to append data to centroids
		// for (std::vector<Point>::iterator it = points->begin();
		// 	 it != points->end(); ++it)
		// {
		// 	int clusterId = it->cluster;
		// 	nPoints[clusterId] += 1;
		// 	sumX[clusterId] += it->x;
		// 	sumY[clusterId] += it->y;

		// 	it->minDist = __DBL_MAX__; // reset distance
		// }

		// Compute the new centroids
		// for (std::vector<Point>::iterator c = begin(centroids);
		// 	 c != end(centroids); ++c)
		// {
		// 	int clusterId = c - begin(centroids);
		// 	c->x = sumX[clusterId] / nPoints[clusterId];
		// 	c->y = sumY[clusterId] / nPoints[clusterId];
		// 	// std::cout << "cluster: " << c->x << ", " << c->y << std::endl;
		// }
	}
}
int main(int argc, char **argv)
{
	auto point = readCsv();
	kMeansClustering(&point, 30, 5);
	saveCsv(&point, "output.csv");

	return 0;
}
