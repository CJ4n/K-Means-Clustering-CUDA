#include <DataPoints.h>

#include <cuda.h>
#include <thrust/reduce.h>

#include "Config.h"
#include "CudaCheckError.h"
#include "Timer.h"

DataPoints *AllocateDataPoints(int num_features, int num_data_points)
{
	if (MEASURE_TIME)
	{
		timer_memory_allocation_gpu->Start();
	}

	DataPoints *point;
	cudaMallocManaged(&point, sizeof(DataPoints));
	cudaCheckError();

	point->num_data_points = num_data_points;
	cudaMallocManaged(&(point->cluster_id_of_point), sizeof(int) * num_data_points);
	cudaCheckError();
	cudaMemset(point->cluster_id_of_point, 0, sizeof(int) * num_data_points);
	cudaCheckError();

	// point->num_features = num_features;
	cudaMallocManaged(&(point->features_array), sizeof(*(point->features_array)) * num_features);
	cudaCheckError();

	for (int feature = 0; feature < num_features; ++feature)
	{
		cudaMallocManaged(&(point->features_array[feature]), sizeof(MyDataType) * point->num_data_points);
		cudaCheckError();
		cudaMemset(point->features_array[feature], 0, sizeof(MyDataType) * point->num_data_points);
		cudaCheckError();
	}
	if (MEASURE_TIME)
	{
		timer_memory_allocation_gpu->Stop();
		timer_memory_allocation_gpu->Elapsed();
	}
	return point;
}

DataPoints *AllocateDataPointsDevice(int num_features, int num_data_points, DataPoints *p)
{
	DataPoints *point;
	cudaMallocManaged(&point, sizeof(DataPoints));
	cudaCheckError();
	point->num_data_points = p->num_data_points;

	cudaMalloc(&(point->cluster_id_of_point), sizeof(int) * num_data_points);
	cudaCheckError();
	cudaMemcpy((void *)(point->cluster_id_of_point), (void *)(p->cluster_id_of_point), sizeof(int) * num_data_points, cudaMemcpyDefault);
	cudaCheckError();

	cudaMallocManaged(&(point->features_array), sizeof(*(point->features_array)) * num_features);
	cudaCheckError();

	for (int feature = 0; feature < num_features; ++feature)
	{
		cudaMalloc(&(point->features_array[feature]), sizeof(MyDataType) * num_data_points);
		cudaCheckError();
		cudaMemcpy((void *)(point->features_array[feature]), (void *)(p->features_array[feature]), sizeof(MyDataType) * num_data_points, cudaMemcpyDefault);
		cudaCheckError();
	}

	return point;
}

void DeallocateDataPoints(DataPoints *data_points, const int num_features)
{
	if (MEASURE_TIME)
	{
		timer_memory_allocation_gpu->Start();
	}
	for (int f = 0; f < num_features; f++)
	{
		cudaFree(data_points->features_array[f]);
	}
	cudaFree(data_points->features_array);
	cudaFree(data_points->cluster_id_of_point);
	cudaFree(data_points);
	if (MEASURE_TIME)
	{
		timer_memory_allocation_gpu->Stop();
		timer_memory_allocation_gpu->Elapsed();
	}
}

MyDataType Distance(const DataPoints *p1, const DataPoints *p2, const int point_id, const int cluster_id, const int num_features)
{
	MyDataType error = 0;
	for (int feature = 0; feature < num_features; ++feature)
	{
		error += (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]) * (p1->features_array[feature][cluster_id] - p2->features_array[feature][point_id]);
	}
	return error;
}

MyDataType MeanSquareError(const DataPoints *point, const DataPoints *centroid, const int num_features)
{
	MyDataType error = 0.0;
	for (int i = 0; i < point->num_data_points; ++i)
	{
		error += Distance(centroid, point, i, point->cluster_id_of_point[i], num_features);
	}
	return error / point->num_data_points;
}

#define INDEX_CLUSTER(c, f, num_clusters) (f * num_clusters + c)
template <int F_NUM>
__global__ void CalculateErrorsForEachPoint(const DataPoints *points, const DataPoints *centroids, MyDataType *sum_erros, const int num_clusters, const int num_points, const int active_threads_count)
{ // acctualy, one thrad calcualtes error for two points
	extern __shared__ MyDataType shm_e1[];
	// const int tid = threadIdx.x;
	const int gid_read = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	const int gid_write = blockIdx.x * blockDim.x + threadIdx.x;

	int c1, c2;

	// shm_e1[INDEX_E1(tid)] = 0;
	if (gid_read >= num_points)
	{
		return;
	}
	// if (tid >= active_threads_count)
	// {
	// 	return;
	// }
	c1 = points->cluster_id_of_point[gid_read];
	if (gid_read + blockDim.x < num_points)
	{
		c2 = points->cluster_id_of_point[gid_read + blockDim.x];
	}

	MyDataType error = 0;
	for (int f = 0; f < F_NUM; ++f)
	{
		MyDataType tmp = centroids->features_array[f][c1] - points->features_array[f][gid_read];

		error += tmp * tmp;
		if (gid_read + blockDim.x < num_points)
		{
			tmp = centroids->features_array[f][c2] - points->features_array[f][gid_read + blockDim.x];
			error += tmp * tmp;
		}
	}

	sum_erros[gid_write] = error;
}

template <int F_NUM>
MyDataType MeanSquareErrorParallel(const DataPoints *points, const DataPoints *centroids)
{
	const int N = points->num_data_points;
	const int num_threads = 1024;
	const int num_blocks = std::ceil(N / num_threads / 2);
	const int num_clusters = centroids->num_data_points;
	const int act = num_threads;
	const size_t shm_e_size = sizeof(MyDataType) * F_NUM * num_clusters;
	MyDataType *errors;

	cudaMallocManaged(&errors, sizeof(MyDataType) * N / 2);
	cudaCheckError();

	CalculateErrorsForEachPoint<F_NUM><<<num_blocks, num_threads, shm_e_size>>>(points, centroids, errors, num_clusters, N, act);
	cudaDeviceSynchronize();
	cudaCheckError();

	MyDataType error = thrust::reduce(errors, errors + N / 2, (MyDataType)0.0);
	cudaDeviceSynchronize();
	cudaCheckError();

	error /= (MyDataType)points->num_data_points;
	cudaFree(errors);
	cudaCheckError();
	return error;
}

template MyDataType MeanSquareErrorParallel<NUM_FEATURES>(const DataPoints *points, const DataPoints *centroids);
