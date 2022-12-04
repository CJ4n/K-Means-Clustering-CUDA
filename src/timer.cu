#include "timer.h"
#include "cudaCheckError.h"

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
      cudaEventRecord(start, 0);
}

void GpuTimer::Stop()
{
      cudaEventRecord(stop, 0);
}

float GpuTimer::Elapsed()
{
      float elapsed;
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed, start, stop);
      total_time += elapsed;
      return elapsed;
}

GpuTimer *timer_find_closest_centroids;
GpuTimer *timer_compute_centroids;
GpuTimer *timer_memory_allocation_gpu;
GpuTimer *timer_gpu_version;
GpuTimer *timer_thurst_version;
GpuTimer *timer_cpu_version;
GpuTimer *timer_data_generations;
