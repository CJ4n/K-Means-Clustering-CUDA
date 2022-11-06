
#define cudaCheckError()                                                                    \
	{                                                                                       \
		cudaError_t e = cudaGetLastError();                                                 \
		if (e != cudaSuccess)                                                               \
		{                                                                                   \
			printf("Cudafailure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0);                                                                        \
		}                                                                                   \
	}
    