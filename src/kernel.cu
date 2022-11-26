

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
#include "timer.h"

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
	if(epoch>0)
		std::cout << "epoch: " << epoch << " Error: " << MeanSquareError(point, centroids) << std::endl;
		k_means_one_iteration_algorithm(point, centroids);
		cudaDeviceSynchronize();

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
	DeallocateDataPoints(point);
}
void InitTimers()
{
      timer_find_closest_centroids = new GpuTimer();
      timer_compute_centroids = new GpuTimer();
      timer_memory_allocation_gpu = new GpuTimer();
      timer_gpu_version = new GpuTimer();
      timer_thurst_version = new GpuTimer();
      timer_cpu_version = new GpuTimer();
}

void DeleteTimers()
{
      delete timer_compute_centroids;
      delete timer_cpu_version;
      delete timer_gpu_version;
      delete timer_thurst_version;
      delete timer_memory_allocation_gpu;
      delete timer_find_closest_centroids;
}

int main(int argc, char **argv)
{
	InitTimers();

	// int num_features = 3;
	// int num_points = 1 << 19;// nadal jest problem z dużymi liczbami
	// int num_cluster = 7;

	// int num_epoches = 5;
	// //________________________________THRUST________________________________
	// std::cout << "----------------THURST----------------" << std::endl;
	// timer_thurst_version->Start();
	// // RunKMeansClustering(KMeansOneIterationGpuThurst, "THRUST", num_features, num_points, num_cluster, num_epoches);
	// timer_thurst_version->Stop();
	// timer_thurst_version->Elapsed();
	// std::cout << "THURST implementation: " << timer_thurst_version->total_time << std::endl;
	// //________________________________THRUST________________________________

	// //__________________________________CPU_________________________________
	// std::cout << "-----------------CPU------------------" << std::endl;
	// timer_cpu_version->Start();
	// RunKMeansClustering(KMeansOneIterationCpu, "CPU", num_features, num_points, num_cluster, num_epoches);
	// timer_cpu_version->Stop();
	// timer_cpu_version->Elapsed();
	// std::cout << "CPU implementation: " << timer_cpu_version->total_time << std::endl;
	// //__________________________________CPU_________________________________


	// //__________________________________GPU_________________________________
	// std::cout << "-----------------GPU------------------" << std::endl;
	// timer_gpu_version->Start();
	// RunKMeansClustering(KMeansOneIterationGpu, "GPU", num_features, num_points, num_cluster, num_epoches);
	// timer_gpu_version->Stop();
	// timer_gpu_version->Elapsed();

	// std::cout << "compute_centroids: " << timer_compute_centroids->total_time << "ms" << std::endl;
	// std::cout << "find_closest_centroids: " << timer_find_closest_centroids->total_time << "ms" << std::endl;
	// std::cout << "GPU implementation: " << timer_gpu_version->total_time << "ms" << std::endl;
	// //__________________________________GPU_________________________________

	
	// save generated points
	// DataPoints *point = GeneratePoints(num_features, num_points);
	// SaveCsv(point, "Input.csv");
	// // DeallocateDataPoints(point);

	for (int num_features = 1; num_features < 5; num_features++)
	for (int num_cluster = 1; num_cluster < 6; num_cluster++)
	{
		int num_points = 1 << 19;
		// int num_cluster = 6;

		int num_epoches = 2;
		std::cout << "features: " << num_features <<", clusters: "<<num_cluster <<std::endl;
		RunKMeansClustering(KMeansOneIterationCpu, "CPU", num_features, num_points, num_cluster, num_epoches);
		RunKMeansClustering(KMeansOneIterationGpu, "GPU", num_features, num_points, num_cluster, num_epoches);
	}
DeleteTimers();
	return 0;
}
