#include <ctime>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <unistd.h>
#include <iomanip>

#include "Constants.h"
#include "cudaCheckError.h"
#include "dataPoints.h"
#include "GeneratePoints.h"
#include "kMeansCpu.h"
#include "kMeansGpuThrust.h"
#include "kMeansGpu.h"
#include "timer.h"

#define RANDOM_CENTROID_INITIALIZATION 0

DataPoints *GetCentroids(DataPoints *point, int num_clusters, int num_features)
{
	DataPoints *centroids = AllocateDataPoints(num_features, num_clusters);

	for (int i = 0; i < num_clusters; ++i)
	{
		// int n = rand() % point->num_data_points;
		for (int feature = 0; feature < num_features; ++feature)
		{
			centroids->features_array[feature][i] = point->features_array[feature][i];
		}

		centroids->cluster_id_of_point[i] = i;
	}
	return centroids;
}
#define DEBUG 0 // set to 1, if you want to run program for many num_cluster and num_points at once

double kMeansClustering(DataPoints *point, int epochs, int num_clusters, int num_features, int num_points, void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *, const int, const int))
{
	DataPoints *centroids = GetCentroids(point, num_clusters, num_features);
	double final_error = 0;
	final_error = MeanSquareError(point, centroids, num_points,  num_features);
	if (!DEBUG)
	{
		std::cout << "EPOCH: " << -1 << " ERROR: " << final_error << std::endl;
	}
	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		k_means_one_iteration_algorithm(point, centroids, num_clusters, num_features);
		cudaDeviceSynchronize();

		cudaCheckError();
		// COMMENT/UNCOMMENT THIS SLEEP
		// sleep(1);
		// final_error = MeanSquareError(point, centroids);
		if (!DEBUG)
		{
			std::cout << "EPOCH: " << epoch << " ERROR: " << final_error << std::endl;
		}
	}

	final_error = MeanSquareError(point, centroids, num_points, num_features);
	if (!DEBUG)
	{
		std::cout << "EPOCH: "
				  << "afer algorithm"
				  << " ERROR: " << final_error << std::endl;
	}
	DeallocateDataPoints(centroids,num_features);
	return final_error;
}

double RunKMeansClustering(void (*k_means_one_iteration_algorithm)(DataPoints *, DataPoints *, const int, const int), std::string alg_name, int num_features, int num_points, int num_cluster, int num_epochs)
{
	std::srand(0);
	DataPoints *point = GeneratePoints(num_features, num_points);
	double error = kMeansClustering(point, num_epochs, num_cluster, num_features, num_points, k_means_one_iteration_algorithm);
	DeallocateDataPoints(point,num_features);
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
	timer_data_generations = new GpuTimer();
}

void DeleteTimers()
{
	delete timer_compute_centroids;
	delete timer_cpu_version;
	delete timer_gpu_version;
	delete timer_thurst_version;
	delete timer_memory_allocation_gpu;
	delete timer_find_closest_centroids;
	delete timer_data_generations;
}

// jeśli sleep w lini 54 jest odkomentowany to wynik jest dobry, jeśli nie to jest losy, choziaż czasem zdarzy się że jest dobry.

// jeśli DEBUG w pliku kMeansGpu.cu w lini 263 jest ustawiony na 1 do też wynik jest dobry, jak go nie ma to wynik jest losowy, podobnie jak z sleep.

// dodanie sleepa tak właściwe gdzie kolwiek w petli w 47 lini, sprawia że wynik jest dobry.

// dodanie sleepa na początku(przed linia 434) lub na końcy pliku(po DeallocateDataPoints(reduced_points); linia 522) kMeansGpu.cu też pomaga, choziaż u mnie jak dam sleepa w lini 433 to pomaga
// do dla kilku pierwszych epok, ale dla ostanij już nie. Ale to czy pomoże też jest często losowy.

// Dla mnie to wygląda jak jakiś race condition między kolejnymi iteraciami algorytmu, tylko nie mam pojęcia dlaczego miałby w ogóle wysępować.

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
		RunKMeansClustering(KMeansOneIterationCpu<NUM_FEATURES>, "CPU", NUM_FEATURES, NUM_POINTS, NUM_CLUSTERS, NUM_EPOCHES);
		timer_cpu_version->Stop();
		timer_cpu_version->Elapsed();
		//__________________________________CPU_________________________________
		timer_data_generations->total_time = 0; // every run of RunKmeanClustering genertates points (I want point to be exacly the same to easier spot bugs)
		//__________________________________GPU_________________________________
		std::cout << "-----------------GPU------------------" << std::endl;
		timer_gpu_version->Start();
		RunKMeansClustering(KMeansOneIterationGpu<NUM_FEATURES>, "GPU", NUM_FEATURES, NUM_POINTS, NUM_CLUSTERS, NUM_EPOCHES);
		timer_gpu_version->Stop();
		timer_gpu_version->Elapsed();
		//__________________________________GPU_________________________________

		std::cout << "THURST implementation:  " << timer_thurst_version->total_time << "ms" << std::endl;

		std::cout << "CPU implementation:     " << timer_cpu_version->total_time << "ms" << std::endl;

		std::cout << "GPU implementation:     " << timer_gpu_version->total_time << "ms" << std::endl;
		std::cout << "compute_centroids:      " << timer_compute_centroids->total_time << "ms" << std::endl;
		std::cout << "find_closest_centroids: " << timer_find_closest_centroids->total_time << "ms" << std::endl;
		std::cout << "memory_allocation_gpu:  " << timer_memory_allocation_gpu->total_time << "ms" << std::endl;
		std::cout << "timer_data_generations: " << timer_data_generations->total_time << "ms" << std::endl;

		// save generated points
		// DataPoints *point = GeneratePoints(num_features, num_points);
		// SaveCsv(point, "Input.csv");
		// // DeallocateDataPoints(point);
	}
	else // test for many combinations of params
	{
		int f = NUM_FEATURES;
		// for (int f = 1; f < 7; f++)
		for (int c = 3; c < 12; c++)
			for (int i = 17; i < 25; i++)
			{
				int num_points = 1 << i;
				// problme bo numfeartes is tempalte a tu f sie zwiskza !!!!
				const double exact_error = RunKMeansClustering(KMeansOneIterationCpu<NUM_FEATURES>, "CPU", f, num_points, c, NUM_EPOCHES);
				const double gpu_error = RunKMeansClustering(KMeansOneIterationGpu<NUM_FEATURES>, "GPU", f, num_points, c, NUM_EPOCHES);
				if (std::abs(exact_error - gpu_error) > 10e-7)
				{
					std::cout << "<<|||||||||||||||||||||||||dfd|||"
							  << "num_cluster: " << c << " num_feature: " << f << " num_points: i<<" << i << "||||||||||||||||||||||||||||" << std::endl;
					std::cout << "exact_error: " << exact_error << std::endl;
					std::cout << "gpu_error:   " << gpu_error << std::endl;
				}
			}
	}
	DeleteTimers();
	return 0;
}
