#include "cuda.h"

class GpuTimer
{
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

void DeleteTimers();
void InitTimers();
extern GpuTimer *timer_find_closest_centroids;
extern GpuTimer *timer_compute_centroids;
extern GpuTimer *timer_data_generation;
extern GpuTimer *timer_gpu_version;
extern GpuTimer *timer_thurst_version;
extern GpuTimer *timer_cpu_version;
extern GpuTimer *timer_memory_allocation_gpu;