

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

void kMeansClustering(DataPoints *point, int epochs, int num_clusters, void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *))
{
	std::srand(0); // need to set the random seed

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
	centroids->num_data_points = num_clusters;

	// alloc cuda memory

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		// saveCsv(point, "train" + std::to_string(epoch) + ".csv");
		// std::cout << "START EPOCH " << epoch << std::endl;
		k_means_one_iteration_algorithm(point, centroids);
		// std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, centroids) << std::endl;
		// for (int feature = 0; feature < point->num_features; ++feature)
		// {
		// 	std::cout << "feature: " << feature << " |";
		// 	for (int c = 0; c < centroids->num_data_points; ++c)
		// 	{
		// 		std::cout << centroids->features_array[feature][c] << ", ";
		// 	}
		// 	std::cout << std::endl;
		// }
		// std::cout << std::endl;
	}
	DeallocateDataPoints(centroids);
}
int main(int argc, char **argv)
{
	DataPoints *point = ReadCsv();
	// std::srand(time(0)); // need to set the random seed
	// std::srand(0); // need to set the random seed

	kMeansClustering(point, 5, 5, KMeansOneIterationGpuThurst);
	SaveCsv(point, "outputthurst.csv");

	DeallocateDataPoints(point);

	std::cout << "----------CPUUUUUUUUUU------\n";
	point = ReadCsv();
	kMeansClustering(point, 5, 5, KMeansOneIterationCpu);

	SaveCsv(point, "outputcpu.csv");
	DeallocateDataPoints(point);

	return 0;
}
