

// #include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <ctime>	// for a random seed
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



void kMeansClustering(dataPoints *point, int epochs, int num_clusters, void (*k_means_one_iteration_algorithm)(dataPoints *, dataPoints *))
{
	dataPoints *centroids = allocate_pt(point->num_features, num_clusters);
	
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
	kMeansClustering(point, 6, 5,	k_means_one_iteration_gpu_thurst);
	cudaFree(point);
	std::cout << "----------------\n";
	point = readCsv();
	kMeansClustering(point, 6, 5, k_means_one_iteration_cpu);

	saveCsv(point, "output.csv");

	return 0;
}
