#include "Timer.h"

#include "CudaCheckError.h"

GpuTimer::GpuTimer()
{
	total_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

GpuTimer::~GpuTimer()
{
	cudaEventDestroy(start);

	cudaEventDestroy(stop);
}

void GpuTimer::Start()
{
	cudaEventRecord(start);
}

void GpuTimer::Stop()
{
	cudaEventRecord(stop);
}

float GpuTimer::Elapsed()
{
	float elapsed;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	total_time += elapsed;
	return elapsed;
}

void InitTimers()
{
	timer_find_closest_centroids = new GpuTimer();
	timer_compute_centroids = new GpuTimer();
	timer_data_generation = new GpuTimer();
	timer_gpu_version = new GpuTimer();
	timer_thurst_version = new GpuTimer();
	timer_cpu_version = new GpuTimer();
	timer_memory_allocation_gpu = new GpuTimer();
}

void DeleteTimers()
{
	delete timer_compute_centroids;
	delete timer_cpu_version;
	delete timer_gpu_version;
	delete timer_thurst_version;
	delete timer_data_generation;
	delete timer_find_closest_centroids;
	delete timer_memory_allocation_gpu;
}

GpuTimer *timer_find_closest_centroids;
GpuTimer *timer_compute_centroids;
GpuTimer *timer_data_generation;
GpuTimer *timer_gpu_version;
GpuTimer *timer_thurst_version;
GpuTimer *timer_cpu_version;
GpuTimer *timer_memory_allocation_gpu;
