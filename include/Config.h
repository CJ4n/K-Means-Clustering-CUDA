#pragma once

// algorithm params
#define NUM_CLUSTERS 3
#define NUM_FEATURES 3
#define NUM_POINTS 1 << 20

#define NUM_EPOCHES 4
#define EPS 1e-1
#define END_AFTER_N_EPOCHES 1 // 1 - ends when diff between two consecutive erros is smaller then EPS, 0 - ends after NUM_EPOCHES were computed

#define MIN_FEATURE_VALUE 0
#define MAX_FEATURE_VALUE 35
// algorithm params

// execution params
#define MEASURE_TIME 0 // if measuring time, then result are incorrect, probaly some data race?
#define RUN_REDUCE_FEATURE_WISE 1
#define SYNCHRONIZE_AFTER_KERNEL_RUN 0

// debug params
#define DEBUG_PROGRAM 0
#define DEBUG_GPU_ITERATION 0 // set to 1, if you want display debuging info, such as: sum of all points feature and cluster wise, number of points belonging to each cluster, at different stages of algorithm
