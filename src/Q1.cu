//============================================================================
// Name        : Q1.cpp
// Author      : 
// Version     :
// Copyright   :
// Description : COM2039 Histogram Coursework
//============================================================================

#include "com2039.hpp"

/// Error Checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/// Loading file
size_t loadSamples(const char* path_to_data_points_file, float** ptr ){
    std::ifstream file (path_to_data_points_file, std::ios::in|std::ios::binary|std::ios::ate);
    std::streampos size_read = file.tellg();
    if (size_read < 0){
        std::cout << "Error reading file " << path_to_data_points_file << std::endl;
        exit(1);
    }
    size_t len_array = size_read/sizeof(float);
    std::cout << "Read :" << size_read << " bytes = " << len_array << " elements." << std::endl;

    char* memblock = new char[size_read];
    file.seekg(0, std::ios::beg);
    file.read (memblock, size_read);    file.close();
    std::cout << "Correctly loaded "<< path_to_data_points_file << std::endl;
    *ptr = (float*)memblock;

    return len_array;
}

/////// Find Maximum
__global__ void maxReduceKernel(float *d_in, size_t len_array){
	//
	// Your code goes here.
	//
	__shared__ float sdata[BLOCK_SIZE];
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;


	if (idx < len_array){
		sdata[tid] = d_in[idx];
	}
	else{
		sdata[tid] = -FLT_MAX;
	}
	__syncthreads();


	for (unsigned int stride = blockDim.x /2; stride > 0; stride >>= 1){
		if (tid < stride){
			sdata[tid] = max(sdata[tid],sdata[tid+stride]);
		}
		__syncthreads();
	}


	if (tid == 0){
		d_in[blockIdx.x] = sdata[0];
	}
}


float findMaxValue(float* samples_h, size_t len_array){
	//
	// Your code goes here.
	//
	float *input_d;
	cudaError_t err;
	

    err = cudaMalloc((void**) &input_d, len_array*sizeof(float));
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error allocating memory for input_d: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	err = cudaMemcpy(input_d, samples_h, len_array*sizeof(float),cudaMemcpyHostToDevice);
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error copying samples_h to input_d: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	size_t len_active_array = len_array;
    int num_blocks = (len_active_array+BLOCK_SIZE-1)/BLOCK_SIZE;
	while(len_active_array > BLOCK_SIZE){
		maxReduceKernel<<<num_blocks, BLOCK_SIZE>>>(input_d, len_active_array);
        len_active_array = num_blocks;
        num_blocks = (len_active_array+BLOCK_SIZE-1)/BLOCK_SIZE;
	}



	maxReduceKernel<<<1, BLOCK_SIZE>>>(input_d, len_active_array);


    float result = 0.0f;
    err = cudaMemcpy(&result, input_d, sizeof(float),cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess ){
        std::cout << "CUDA Error copying input_d to result: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	cudaFree(input_d);

	return result;
}


/////// Find Minimum
__global__ void minReduceKernel(float *d_in, size_t len_array){
	//
	// Your code goes here
	//
	__shared__ float sdata[BLOCK_SIZE];
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;


	if (idx < len_array){
		sdata[tid] = d_in[idx];
	}
	else{
		sdata[tid] = FLT_MAX;
	}
	__syncthreads();


	for (unsigned int stride = blockDim.x /2; stride > 0; stride >>= 1){
		if (tid < stride){
			sdata[tid] = min(sdata[tid],sdata[tid+stride]);
		}
		__syncthreads();
	}


	if (tid == 0){
		d_in[blockIdx.x] = sdata[0];
	}
}


float findMinValue(float* samples_h, size_t len_array){
	//
	// Your code goes here
	//
	float *input_d;
	cudaError_t err;
	

    err = cudaMalloc((void**) &input_d, len_array*sizeof(float));
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error allocating memory for input_d: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	err = cudaMemcpy(input_d, samples_h, len_array*sizeof(float),cudaMemcpyHostToDevice);
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error copying samples_h to input_d: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	size_t len_active_array = len_array;
    int num_blocks = (len_active_array+BLOCK_SIZE-1)/BLOCK_SIZE;
	while(len_active_array > BLOCK_SIZE){
		minReduceKernel<<<num_blocks, BLOCK_SIZE>>>(input_d, len_active_array);
        len_active_array = num_blocks;
        num_blocks = (len_active_array+BLOCK_SIZE-1)/BLOCK_SIZE;
	}



	minReduceKernel<<<1, BLOCK_SIZE>>>(input_d, len_active_array);


    float result = 0.0f;
    err = cudaMemcpy(&result, input_d, sizeof(float),cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess ){
        std::cout << "CUDA Error copying input_d to result: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	cudaFree(input_d);

	return result;
}



/////// Create Histogram
__global__ void histogramKernel512(float *d_in, unsigned int *hist, size_t len_array, float min_value, float max_value) {
	//
	// Your code goes here
	//
	__shared__ unsigned int local_hist[512];
	for (size_t bin = threadIdx.x; bin < 512; bin += blockDim.x){
        local_hist[bin] = 0u;
    }
    __syncthreads();


	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_array) {
		float val = d_in[tid];
		float range = max_value - min_value;
        size_t bin = (size_t)((val-min_value)/range *512.0f);
		if (bin >= 512){
			bin = 511;
		}
        atomicAdd( &(local_hist[bin]), 1);
    }
    __syncthreads();


    for (size_t bin = threadIdx.x; bin < 512; bin += blockDim.x){
        unsigned int binValue = local_hist[bin];
        if(binValue>0){
            atomicAdd( &(hist[bin]),binValue);
        }
    }
}



/// histogram
void histogram512(float *samples_h, size_t len_array, unsigned int **hist_h, float min_value, float max_value) {
	//
	// Your code goes here
	//

	float *d_in;
	unsigned int *hist_d;
	cudaError_t err;

	//allocate input mem
    err =  cudaMalloc((void**)&d_in, len_array*sizeof(unsigned int));
    if ( err != cudaSuccess ){
       std::cout << "CUDA Error allocating memory for input_d: " <<  cudaGetErrorString(err) << std::endl;
       exit(-1);
    }

	//allocate output hist mem
    err =  cudaMalloc((void**) &hist_d, 512*sizeof(unsigned int));
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error allocating memory for hist_d: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	err = cudaMemset(hist_d, 0, 512*sizeof(unsigned int));
    if ( err != cudaSuccess ){
            std::cout << "CUDA Error setting values of hist_d to zero: " <<  cudaGetErrorString(err) << std::endl;
            exit(-1);
    }


	err = cudaMemcpy(d_in, samples_h, len_array*sizeof(unsigned int),cudaMemcpyHostToDevice);
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error copying samples_h to input_d: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	dim3 grid_size = (len_array + BLOCK_SIZE -1)/BLOCK_SIZE;
	histogramKernel512<<<grid_size, BLOCK_SIZE>>>(d_in, hist_d, len_array, min_value, max_value);


	err = cudaMemcpy(*hist_h, hist_d, 512*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error copying hist_d back into hist_h: " <<  cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


	cudaFree(hist_d);
    cudaFree(d_in);
}
