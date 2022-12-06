#include <cuda.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "Config.h"
#include "CudaCheckError.h"
#include "DataPoints.h"
#include "GeneratePoints.h"
#include "KMeansCpu.h"
#include "KMeansGpu.h"
#include "KMeansGpuThrust.h"
#include "Timer.h"

DataPoints *GetCentroids(DataPoints *point, int num_clusters, const int num_features)
{
	DataPoints *centroids = AllocateDataPoints(num_features, num_clusters);

	for (int i = 0; i < num_clusters; ++i)
	{
		// int n = rand() % point->num_data_points;
		for (int feature = 0; feature < num_features; ++feature)
		{
			centroids->features_array[feature][i] = point->features_array[feature][i];
			cudaCheckError();
		}
		centroids->cluster_id_of_point[i] = i;
		cudaCheckError();
		// centroids->cluster_id_of_point[i] = i;
	}

	return centroids;
}

double kMeansClustering(DataPoints *point, const int num_clusters, MyDataType (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *))
{
	DataPoints *centroids = GetCentroids(point, num_clusters, NUM_FEATURES);
	MyDataType error = 0;
	MyDataType last_error = 0;
	int epoch = 0;

	while (1)
	{
		error = k_means_one_iteration_algorithm(point, centroids);
		cudaDeviceSynchronize();
		cudaCheckError();
		if (!DEBUG_PROGRAM)
		{
			std::cout << "EPOCH: " << epoch << " ERROR: " << error << std::endl;
		}

		epoch++;

		if (END_AFTER_N_EPOCHES)
		{
			if (epoch >= NUM_EPOCHES)
			{
				break;
			}
		}
		else
		{
			if (epoch == 0)
			{
				last_error = error;
			}
			else
			{
				if (std::abs(last_error - error) < EPS)
				{
					std::cout << "Diff between last error and currnet is closer then " << EPS << ", so ending computation";
					break;
				}
				last_error = error;
			}
		}
	}
	DeallocateDataPoints(centroids, NUM_FEATURES);
	return error;
}

double RunKMeansClustering(MyDataType (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *), std::string alg_name,
						   const int num_points, const int num_cluster, GpuTimer *timer)
{
	timer_memory_allocation_gpu->total_time = 0;
	std::srand(0);

	DataPoints *point = GeneratePoints(NUM_FEATURES, num_points);

	timer->Start();
	double error = kMeansClustering(point, num_cluster, k_means_one_iteration_algorithm);
	timer->Stop();
	timer->Elapsed();
	DeallocateDataPoints(point, NUM_FEATURES);
	return error;
}

int main(int argc, char **argv)
{
	InitTimers();

	std::cout << std::setprecision(10);
	if (!DEBUG_PROGRAM)
	{
		//________________________________THRUST________________________________
		std::cout << "----------------THURST----------------" << std::endl;
		RunKMeansClustering(KMeansOneIterationGpuThurst<NUM_FEATURES>, "THRUST", NUM_POINTS, NUM_CLUSTERS, timer_thurst_version);
		//________________________________THRUST________________________________

		//__________________________________CPU_________________________________
		std::cout << "-----------------CPU------------------" << std::endl;
		RunKMeansClustering(KMeansOneIterationCpu<NUM_FEATURES>, "CPU", NUM_POINTS, NUM_CLUSTERS, timer_cpu_version);
		//__________________________________CPU_________________________________

		//__________________________________GPU_________________________________
		std::cout << "-----------------GPU------------------" << std::endl;
		RunKMeansClustering(KMeansOneIterationGpu<NUM_FEATURES>, "GPU", NUM_POINTS, NUM_CLUSTERS, timer_gpu_version);
		//__________________________________GPU_________________________________

		std::cout << "THURST implementation:  " << timer_thurst_version->total_time << "ms" << std::endl;

		std::cout << "CPU implementation:     " << timer_cpu_version->total_time << "ms" << std::endl;
		std::cout << "GPU implementation:     " << timer_gpu_version->total_time << "ms" << std::endl;
		if (MEASURE_TIME)
		{
			std::cout << "compute_centroids:      " << timer_compute_centroids->total_time << "ms" << std::endl;
			std::cout << "find_closest_centroids: " << timer_find_closest_centroids->total_time << "ms" << std::endl;
			std::cout << "timer_memory_allocation_gpu: " << timer_memory_allocation_gpu->total_time << "ms" << std::endl;
			std::cout << "timer_data_generation: " << timer_data_generation->total_time << "ms" << std::endl;
		}
	}
	else // test for many combinations of params
	{
		for (int c = 3; c < 10; c++)
		{
			for (int i = 17; i < 25; i++)
			{
				std::cout << "num_cluster: " << c << " num_feature: " << NUM_FEATURES << " num_points: i<<" << i << std::endl;
				int num_points = 1 << i;
				const MyDataType exact_error = RunKMeansClustering(KMeansOneIterationCpu<NUM_FEATURES>, "CPU", num_points, c, timer_cpu_version);
				const MyDataType gpu_error = RunKMeansClustering(KMeansOneIterationGpu<NUM_FEATURES>, "GPU", num_points, c, timer_gpu_version);
				// const MyDataType thurst_error = RunKMeansClustering(KMeansOneIterationGpuThurst<NUM_FEATURES>, "THURST", num_points, c, timer_thurst_version);

				if (std::abs(exact_error - gpu_error) > 10e-7 /*|| std::abs(exact_error - thurst_error) > 10e-7*/)
				{
					std::cout << "<<||||||||||||||||||||||||||||"
							  << "num_cluster: " << c << " num_feature: " << NUM_FEATURES << " num_points: i<<" << i << "||||||||||||||||||||||||||||" << std::endl;
					std::cout << "exact_error: " << exact_error << std::endl;
					std::cout << "gpu_error:   " << gpu_error << std::endl;
					// std::cout << "thrust_error:   " << thurst_error << std::endl;
				}
			}
		}
	}

	DeleteTimers();
	return 0;
}
