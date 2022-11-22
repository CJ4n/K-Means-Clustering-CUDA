

// #include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <ctime> // for a random seed
// #include <fstream>	// for file-reading
#include <iostream> // for file-reading
// #include <sstream>	// for file-reading
#include <vector>
// #include <math.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/reduce.h>
#include "dataPoints.h"

#include "cudaCheckError.h"
#include "kMeansCpu.h"
#include "kMeansGpuThrust.h"
#include "kMeansGpu.h"
#include "GeneratePoints.h"

#define DEBUG 0
#define RANDOM_CENTROID_INITIALIZATION 0

DataPoints *GetCentroids(DataPoints *point, int num_clusters)
{
	DataPoints *centroids = AllocateDataPoints(point->num_features, num_clusters);

	for (int i = 0; i < num_clusters; ++i)
	{
		int n = rand() % point->num_data_points;
		for (int feature = 0; feature < point->num_features; ++feature)
		{
			centroids->features_array[feature][i] = point->features_array[feature][i];
		}

		centroids->cluster_id_of_point[i] = i;
	}
	return centroids;
}

void kMeansClustering(DataPoints *point, int epochs, int num_clusters, void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *))
{
	DataPoints *centroids = GetCentroids(point, num_clusters);

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		// saveCsv(point, "train" + std::to_string(epoch) + ".csv");
		if (DEBUG)
		{
			std::cout << "START EPOCH " << epoch << std::endl;
		}
		
		k_means_one_iteration_algorithm(point, centroids);
		cudaDeviceSynchronize();

		std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, centroids) << std::endl;
		if (DEBUG)
		{
			for (int feature = 0; feature < point->num_features; ++feature)
			{
				std::cout << "feature: " << feature << " |";
				for (int c = 0; c < centroids->num_data_points; ++c)
				{
					std::cout << centroids->features_array[feature][c] << ", ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
	DeallocateDataPoints(centroids);
}

void RunKMeansClustering(void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *), std::string alg_name, int num_features, int num_points, int num_cluster, int num_epochs)
{
	std::srand(0);
	DataPoints *point = GeneratePoints(num_features, num_points);
	std::cout << "----------" + alg_name + "----------\n";
	kMeansClustering(point, num_epochs, num_cluster, k_means_one_iteration_algorithm);
	// SaveCsv(point, "Output" + alg_name + ".csv");
	// DeallocateDataPoints(point);
}

int main(int argc, char **argv)
{
	int num_features = 2;
	int num_points = 1 << 15;
	int num_cluster = 7;
	int num_epoches = 10;
	// RunKMeansClustering(KMeansOneIterationGpuThurst, "THRUST", num_features, num_points, num_cluster, num_epoches);
	RunKMeansClustering(KMeansOneIterationCpu, "CPU", num_features, num_points, num_cluster, num_epoches);
	RunKMeansClustering(KMeansOneIterationGpu, "GPU", num_features, num_points, num_cluster, num_epoches);
	DataPoints *point = GeneratePoints(num_features, num_points);
	SaveCsv(point, "Input.csv");
	DeallocateDataPoints(point);
	return 0;
}
