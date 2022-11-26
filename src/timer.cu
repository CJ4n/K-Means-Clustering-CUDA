#include "timer.h"
#include "cudaCheckError.h"
// struct GpuTimer
// {
//       cudaEvent_t start;
//       cudaEvent_t stop;
//       float total_time;
GpuTimer::GpuTimer()
{
      total_time = 0;
      cudaEventCreate(&start);
      cudaCheckError();

      cudaEventCreate(&stop);
      cudaCheckError();
}

GpuTimer::~GpuTimer()
{
      cudaEventDestroy(start);
      cudaCheckError();

      cudaEventDestroy(stop);
      cudaCheckError();
}

void GpuTimer::Start()
{
      cudaEventRecord(start, 0);
      cudaCheckError();
}

void GpuTimer::Stop()
{
      cudaEventRecord(stop, 0);
      cudaCheckError();
}

float GpuTimer::Elapsed()
{
      float elapsed;
      cudaEventSynchronize(stop);
      cudaCheckError();
      cudaEventElapsedTime(&elapsed, start, stop);
      cudaCheckError();

      total_time += elapsed;
      return elapsed;
}
// };

GpuTimer *timer_find_closest_centroids;
GpuTimer *timer_compute_centroids;
GpuTimer *timer_memory_allocation_gpu;
GpuTimer *timer_gpu_version;
GpuTimer *timer_thurst_version;
GpuTimer *timer_cpu_version;

//  GpuTimer timer_closest_centroids;
//  GpuTimer timer_compute_centroids;
//  GpuTimer timer_memory_allocation;
//  GpuTimer timer_gpu_version;
//  GpuTimer timer_thurst_version;
//  GpuTimer timer_cpu_version;
