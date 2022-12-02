#include "cuda.h"

class GpuTimer
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
private:
      cudaEvent_t start;
      cudaEvent_t stop;

public:
      float total_time;
      GpuTimer();

      ~GpuTimer();

      void Start();

      void Stop();

      float Elapsed();
};
extern GpuTimer *timer_find_closest_centroids;
extern GpuTimer *timer_compute_centroids;
extern GpuTimer *timer_memory_allocation_gpu;
extern GpuTimer *timer_gpu_version;
extern GpuTimer *timer_thurst_version;
extern GpuTimer *timer_cpu_version;
