#include <ctime>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "Constants.h"
#include "cudaCheckError.h"
#include "dataPoints.h"
#include "GeneratePoints.h"
#include "kMeansCpu.h"
#include "kMeansGpuThrust.h"
#include "kMeansGpu.h"
#include "timer.h"

#define RANDOM_CENTROID_INITIALIZATION 0

DataPoints *GetCentroids(DataPoints *point, int num_clusters)
{
	DataPoints *centroids = AllocateDataPoints(point->num_features, num_clusters);

	for (int i = 0; i < num_clusters; ++i)
	{
		// int n = rand() % point->num_data_points;
		for (int feature = 0; feature < point->num_features; ++feature)
		{
			centroids->features_array[feature][i] = point->features_array[feature][i];
		}

		centroids->cluster_id_of_point[i] = i;
	}
	return centroids;
}
#define DEBUG 0

double kMeansClustering(DataPoints *point, int epochs, int num_clusters, void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *))
{
	DataPoints *centroids = GetCentroids(point, num_clusters);
	double final_error = 0;
	final_error = MeanSquareError(point, centroids);
	if (!DEBUG)
	{
		std::cout << "EPOCH: " << -1 << " ERROR: " << final_error << std::endl;
	}
	for (int epoch = 0; epoch < epochs; ++epoch)
	{

		k_means_one_iteration_algorithm(point, centroids);
		cudaDeviceSynchronize();
		final_error = MeanSquareError(point, centroids);
		if (!DEBUG)
		{
			std::cout << "EPOCH: " << epoch << " ERROR: " << final_error << std::endl;
		}
	}
	DeallocateDataPoints(centroids);
	return final_error;
}

double RunKMeansClustering(void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *), std::string alg_name, int num_features, int num_points, int num_cluster, int num_epochs)
{
	std::srand(0);
	DataPoints *point = GeneratePoints(num_features, num_points);
	double error = kMeansClustering(point, num_epochs, num_cluster, k_means_one_iteration_algorithm);
	DeallocateDataPoints(point);
	return error;
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
#include <iomanip>

// TODO: zmusic do dzialanie reduce by feature
int main(int argc, char **argv)
{

	std::cout << std::setprecision(15);
	InitTimers();
	if (!DEBUG)
	{
		//________________________________THRUST________________________________
		std::cout << "----------------THURST----------------" << std::endl;
		timer_thurst_version->Start();
		// RunKMeansClustering(KMeansOneIterationGpuThurst, "THRUST", constants::num_features, constants::num_points, constants::num_cluster, constants::num_epoches);
		timer_thurst_version->Stop();
		timer_thurst_version->Elapsed();
		//________________________________THRUST________________________________

		//__________________________________CPU_________________________________
		std::cout << "-----------------CPU------------------" << std::endl;
		timer_cpu_version->Start();
		RunKMeansClustering(KMeansOneIterationCpu, "CPU", constants::num_features, constants::num_points, constants::num_cluster, constants::num_epoches);
		timer_cpu_version->Stop();
		timer_cpu_version->Elapsed();
		//__________________________________CPU_________________________________

		//__________________________________GPU_________________________________
		std::cout << "-----------------GPU------------------" << std::endl;
		timer_gpu_version->Start();
		RunKMeansClustering(KMeansOneIterationGpu, "GPU", constants::num_features, constants::num_points, constants::num_cluster, constants::num_epoches);
		timer_gpu_version->Stop();
		timer_gpu_version->Elapsed();
		//__________________________________GPU_________________________________

		std::cout << "THURST implementation:  " << timer_thurst_version->total_time << "ms" << std::endl;

		std::cout << "CPU implementation:     " << timer_cpu_version->total_time << "ms" << std::endl;

		std::cout << "GPU implementation:     " << timer_gpu_version->total_time << "ms" << std::endl;
		std::cout << "compute_centroids:      " << timer_compute_centroids->total_time << "ms" << std::endl;
		std::cout << "find_closest_centroids: " << timer_find_closest_centroids->total_time << "ms" << std::endl;

		// save generated points
		// DataPoints *point = GeneratePoints(num_features, num_points);
		// SaveCsv(point, "Input.csv");
		// // DeallocateDataPoints(point);
	}
	else
	{
		for (constants::num_features = 1; constants::num_features < 7; constants::num_features++)
			for (constants::num_cluster = 3; constants::num_cluster < 7; constants::num_cluster++)
				for (int i = 17; i < 22; i++)
				{
					constants::num_points = 1 << i;

					const double exact_error = RunKMeansClustering(KMeansOneIterationCpu, "CPU", constants::num_features, constants::num_points, constants::num_cluster, constants::num_epoches);
					const double gpu_error = RunKMeansClustering(KMeansOneIterationGpu, "GPU", constants::num_features, constants::num_points, constants::num_cluster, constants::num_epoches);
					if (std::abs(exact_error - gpu_error) > 10e-7)
					{
						std::cout << "<<|||||||||||||||||||||||||dfd|||"
								  << "num_cluster: " << constants::num_cluster << " num_feature: " << constants::num_features << " num_points: i<<" << i << "||||||||||||||||||||||||||||" << std::endl;
						std::cout << "exact_error: " << exact_error << std::endl;
						std::cout << "gpu_error:   " << gpu_error << std::endl;
					}
				}
	}
	DeleteTimers();
	return 0;
}
