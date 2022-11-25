#include "cuda.h"

struct GpuTimer
{

//  cudaEvent_t start;
      // cudaEvent_t stop;
      // float total_time;
      // GpuTimer()
      // {
      //       total_time = 0;
      //       cudaEventCreate(&start);
      //       cudaEventCreate(&stop);
      // }

      // ~GpuTimer()
      // {
      //       cudaEventDestroy(start);
      //       cudaEventDestroy(stop);
      // }

      // void Start()
      // {
      //       cudaEventRecord(start, 0);
      // }

      // void Stop()
      // {
      //       cudaEventRecord(stop, 0);
      // }

      // float Elapsed()
      // {
      //       float elapsed;
      //       cudaEventSynchronize(stop);
      //       cudaEventElapsedTime(&elapsed, start, stop);
      //       total_time += elapsed;
      //       return elapsed;
      // }
      cudaEvent_t start;
      cudaEvent_t stop;
      float total_time;
      GpuTimer();

      ~GpuTimer();

      void Start();

      void Stop();

      float Elapsed();
};
extern GpuTimer timer_find_closest_centroids;
extern GpuTimer timer_compute_centroids;
extern GpuTimer timer_memory_allocation;
extern GpuTimer timer_gpu_version;
extern GpuTimer timer_thurst_version;
extern GpuTimer timer_cpu_version;

// static void InitTimers()
// {
//       // timer_closest_centroids = new GpuTimer();
//       // timer_compute_centroids = new GpuTimer();
//       // timer_memory_allocation = new GpuTimer();
//       // timer_gpu_version = new GpuTimer();
//       // timer_thurst_version = new GpuTimer();
//       // timer_cpu_version = new GpuTimer();
// }
