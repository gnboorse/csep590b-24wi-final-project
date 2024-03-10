#include <wb.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <time.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_SIZE 128 // TODO: change me?
#define QUERY_BATCH 512
#define QUERY_LENGTH 2000 // TODO: change me?
#define ITEMS_PER_THREAD 2 // TODO: change me?
#define MIN_RANDOM_VAL 58
#define MAX_RANDOM_VAL 120

__global__ void calculate_mean_and_variance(float *inputArray, float *outputSums, float *outputSquaredSums, int inputLength) {
    __shared__ int sums[BLOCK_SIZE];
    __shared__ int squaredSums[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int j = i + blockDim.x;
    if (j < inputLength) {
        // pre-add 2 numbers together to store in shared memory
        float input1 = inputArray[i];
        float input2 = inputArray[j];
        sums[tid] = input1 + input2;
        squaredSums[tid] = input1 * input1 + input2 * input2;
    } else if (i < inputLength) {
        // boundary condition for cases where the 2nd number to add is out of bounds
        float input1 = inputArray[i];
        sums[tid] = input1;
        squaredSums[tid] = input1 * input1;
    } else {
        // true boundary condition 
        sums[tid] = 0; // is it required to initialize shared memory to zero?
        squaredSums[tid] = 0;
    }
    
    __syncthreads();

    // Traverse the reduction tree
    for (int s = blockDim.x/2; s > 0; s>>=1) {
        if (tid<s) {
            sums[tid] += sums[tid + s];
            squaredSums[tid] += squaredSums[tid + s];
            if (s == 1) {
                outputSums[blockIdx.x] = sums[tid];
                outputSquaredSums[blockIdx.x] = squaredSums[tid];   
            }
        }
        __syncthreads();
    } 
}

__constant__ float mean_X;
__constant__ float variance_Y;

__global__ void normalize_data(float *inputArray, float *outputArray, int inputLength) {
    int i = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;

    // update each value based on mean and variance
    // do this ITEMS_PER_THREAD times
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int jOffset = i + (j * blockDim.x);
        if (jOffset < inputLength) {
            float inputVal = inputArray[jOffset];
            outputArray[jOffset] = (inputVal - mean_X) * variance_Y; // (input[i] - X) / Y
        }
    }
}

__global__ void dtw_matrix(float *A, float *B, float *C, 
                              int queryLength, int referenceLength,
                              int DTWRows, int DTWColumns) {
  int rowid = blockDim.y * blockIdx.y + threadIdx.y; 
  int colid = blockDim.x * blockIdx.x + threadIdx.x; 

  if((rowid < DTWRows) && (colid < DTWColumns)) {
    float cost = 0.0f;
    //No. of diagonals in a  M*N matrix is M+N-1
    for(int digIndex=0; digIndex < DTWRows + DTWColumns - 1; digIndex++) {
      //Diagonal items can be found by rowindex+colindex
      if((rowid+colid) == digIndex) {
        cost = A[rowid] - B[colid];
          if (rowid == 0 && colid==0) { //if col=0 & row=0 nothing
            cost = (cost*cost);
          } else if (colid == 0) { //if col=0 - no left & no topleft
            cost = (cost*cost) + C[((rowid-1)*DTWColumns) + colid];
          } else if (rowid == 0) { //if row=0 - no top & no topleft
            cost = (cost*cost) + C[(rowid*DTWColumns) + (colid-1)];
          } else {
            cost = (cost*cost) + fminf(
              C[((rowid-1)*DTWColumns) + (colid-1)], //topLeft
              fminf(C[((rowid-1)*DTWColumns) + (colid)], //top
              C[(rowid*DTWColumns) + (colid-1)]) //left
             );
          }
        C[(rowid*DTWColumns) + colid] = cost;       
      }
      __syncthreads();
    }
  }
}


int main(int argc, char **argv) {
    srand(time(NULL));
    float *hostNormalizedQuery; // the reference/query string we want to normalize
    float *hostBatchNormalizerInput;
    float *hostNormalizerOutput; // the result of normalizing the string
    float *hostBatchNormalizedQuery;
    float *hostOutputSums; // summation of all items in input
    float *hostOutputSquaredSums; // sum of all squares in input (used for variance calculation)
    float *hostReference;
    float *hostDTW; // The output DTW matrix

    float *deviceNormalizerInput;
    float *deviceNormalizerOutput;
    float *deviceOutputSums;
    float *deviceOutputSquaredSums;
    float *deviceQuery;
    float *deviceReference;
    float *deviceDTW;

    wbTime_start(Generic, "Importing data and creating memory on host");
    // import the query file
    int queryLength, numQuerys;
    hostBatchNormalizerInput = (float *)wbImport(wbPath_join(wbDirectory_current(), "query-string", "query.raw"), &numQuerys, &queryLength);
    wbLog(TRACE, "Number of Queries : ", numQuerys, " Size of each Query : ", queryLength);
    // import the reference file file
    int referenceLength;
    hostReference = (float *)wbImport(wbPath_join(wbDirectory_current(), "reference-string", "normalized-reference-string.raw"), &referenceLength);
    wbLog(TRACE, "Reference string length : ", referenceLength);
     
    int numBlockSums = queryLength / (BLOCK_SIZE << 1);
    if (queryLength % (BLOCK_SIZE << 1)) {
        numBlockSums++;
    }
    
    int inputSizeBytes = queryLength * sizeof(float);
    int blockSumsSizeBytes = numBlockSums * sizeof(float);
    int batchSizeBytes = queryLength * numQuerys * sizeof(float);

    // allocate host memory
    hostOutputSums = (float *)malloc(blockSumsSizeBytes);
    hostOutputSquaredSums = (float *)malloc(blockSumsSizeBytes);
    hostNormalizerOutput = (float *)malloc(inputSizeBytes);
    hostNormalizedQuery = (float *)malloc(inputSizeBytes);
    hostBatchNormalizedQuery = (float *)malloc(batchSizeBytes);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbTime_start(GPU, "Allocating GPU memory.");
    // allocate device memory
    wbCheck(hipMalloc(&deviceNormalizerInput, inputSizeBytes));
    wbCheck(hipMalloc(&deviceNormalizerOutput, inputSizeBytes));
    wbCheck(hipMalloc(&deviceOutputSums, blockSumsSizeBytes));
    wbCheck(hipMalloc(&deviceOutputSquaredSums, blockSumsSizeBytes));
    wbTime_stop(GPU, "Allocating GPU memory.");

    //--Start processing each query for normalizing
    for(int currentQuery = 0; currentQuery < numQuerys; currentQuery++) {
        for(int i = 0; i < queryLength; i++) {
            hostNormalizedQuery[i] = hostBatchNormalizerInput[(currentQuery * queryLength) +i];
        }
        wbTime_start(GPU, "Copying input memory to the GPU.");
        // copy input to GPU
        wbCheck(hipMemcpy(deviceNormalizerInput, hostNormalizedQuery, inputSizeBytes, hipMemcpyHostToDevice));
        wbTime_stop(GPU, "Copying input memory to the GPU.");

        // initialize grid/block dimensions for the kernel that calculates sums
        dim3 gridDimensions_sums(numBlockSums, 1, 1);
        dim3 blockDimensions_sums(BLOCK_SIZE, 1, 1);

        wbTime_start(Compute, "Performing HIP computation");
        // calculate all sums and squared sums
        hipLaunchKernelGGL(
            calculate_mean_and_variance, gridDimensions_sums, blockDimensions_sums, 0, 0, 
            deviceNormalizerInput, deviceOutputSums, deviceOutputSquaredSums, queryLength
        );
        hipDeviceSynchronize();
        wbTime_stop(Compute, "Performing HIP computation");
                
        // Copy the GPU memory back to the CPU here
        wbTime_start(Copy, "Copying output memory to the CPU");
        wbCheck(hipMemcpy(hostOutputSums, deviceOutputSums, blockSumsSizeBytes, hipMemcpyDeviceToHost));
        wbCheck(hipMemcpy(hostOutputSquaredSums, deviceOutputSquaredSums, blockSumsSizeBytes, hipMemcpyDeviceToHost));
        wbTime_stop(Copy, "Copying output memory to the CPU");
        // reduce output sums & squared sums on the host
        for (unsigned int ii = 1; ii < numBlockSums; ii++) {
            hostOutputSums[0] += hostOutputSums[ii];
            hostOutputSquaredSums[0] += hostOutputSquaredSums[ii];
        }
        float mean = hostOutputSums[0] / queryLength;
        float variance = hostOutputSquaredSums[0];
        variance  = variance/queryLength- mean*mean;
        // this is 1 / sigma since we need to divide by sigma elsewhere.
        // Faster to multiply by this factor than to compute lots of division on the GPU
        variance  = variance > 0 ? 1.0/std::sqrt(variance) : 1;
        //wbLog(TRACE, "mu: ", mean, ", sigma: ", 1 / variance);

        // copy reduced sums to constant memory on the device
        wbCheck(hipMemcpyToSymbol(mean_X, &mean, sizeof(float), 0, hipMemcpyHostToDevice));
        wbCheck(hipMemcpyToSymbol(variance_Y, &variance, sizeof(float), 0, hipMemcpyHostToDevice));

        // initialize grid/block dimensions for the kernel that calculates sums
        int numNormalizeThreadsPerBlock = queryLength / (BLOCK_SIZE * ITEMS_PER_THREAD);
        if (queryLength % (BLOCK_SIZE * ITEMS_PER_THREAD)) {
            numNormalizeThreadsPerBlock++;
        }
        dim3 gridDimensions_normalize(numNormalizeThreadsPerBlock, 1, 1);
        dim3 blockDimensions_normalize(BLOCK_SIZE, 1, 1);

        wbTime_start(Compute, "Performing HIP computation");
        // calculate all sums and squared sums
        hipLaunchKernelGGL(
            normalize_data, gridDimensions_normalize, blockDimensions_normalize, 0, 0, 
            deviceNormalizerInput, deviceNormalizerOutput, queryLength
        );
        hipDeviceSynchronize();
        wbTime_stop(Compute, "Performing HIP computation");
        
        // Copy the device normalized array back to the host 
        wbTime_start(Copy, "Copying data from the GPU");
        wbCheck(hipMemcpy(hostNormalizerOutput, deviceNormalizerOutput, inputSizeBytes, hipMemcpyDeviceToHost));
        wbTime_stop(Copy, "Copying data from the GPU");

        for(int i = 0; i < queryLength; i++) {
            hostBatchNormalizedQuery[(currentQuery * queryLength) + i] = hostNormalizerOutput[i];
        }
    }
    
    //Freeing up the device memory used for normalization.
    hipFree(deviceNormalizerInput);
    hipFree(deviceNormalizerOutput);
    hipFree(deviceOutputSums);
    hipFree(deviceOutputSquaredSums);

    //--Start processing each query for creating DTW matrix
    int DTWRows = queryLength;
    int DTWColumns = referenceLength;
    //@@ Allocate the hostDTW matrix
    hostDTW = (float *)malloc(DTWRows * DTWColumns * sizeof(float));
    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    hipMalloc(&deviceQuery, queryLength * sizeof(float));
    hipMalloc(&deviceReference, referenceLength * sizeof(float));
    hipMalloc(&deviceDTW, DTWRows * DTWColumns * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");
    //Since reference is constant, we only need to copy to device only once.
    hipMemcpy(deviceReference, hostReference, referenceLength * sizeof(float), hipMemcpyHostToDevice);
    //@@ Initialize the grid and block dimensions here
    unsigned int blockSize = 32;
    dim3 dtw_threads(blockSize,blockSize,1);
    dim3 dtw_blocks((DTWColumns+blockSize-1)/blockSize, (DTWRows+blockSize-1)/blockSize, 1);

    for(int currentQuery = 0; currentQuery < numQuerys; currentQuery++) {
      for(int i = 0; i < queryLength; i++) {
        hostNormalizedQuery[i] = hostBatchNormalizedQuery[(currentQuery * queryLength) +i];
      }
      wbTime_start(GPU, "Copying input memory to the GPU.");
      //@@ Copy memory to the GPU here
      hipMemcpy(deviceQuery, hostNormalizedQuery, queryLength * sizeof(float), hipMemcpyHostToDevice);
      wbTime_stop(GPU, "Copying input memory to the GPU.");

      wbTime_start(Compute, "Performing HIP computation");
      //@@ Launch the GPU Kernel here
      hipLaunchKernelGGL(dtw_matrix, dtw_blocks, dtw_threads, 0, 0, deviceQuery, deviceReference, deviceDTW, queryLength,
        referenceLength, DTWRows, DTWColumns);
      
      hipDeviceSynchronize();
      wbTime_stop(Compute, "Performing HIP computation");

      wbTime_start(Copy, "Copying output memory to the CPU");
      //@@ Copy the GPU memory back to the CPU here
      hipMemcpy(hostDTW, deviceDTW, DTWRows * DTWColumns * sizeof(float), hipMemcpyDeviceToHost);
      wbTime_stop(Copy, "Copying output memory to the CPU");
    }

    // write output to raw file
    //wbExport(wbPath_join(wbDirectory_current(), "query-string", "normalized_query.raw"), (wbReal_t *)hostBatchNormalizedQuery, numQuerys, queryLength);
    // Free device memory
    
    hipFree(deviceQuery);
    hipFree(deviceReference);
    hipFree(deviceDTW);

    // Free host memory
    free(hostNormalizedQuery);
    free(hostBatchNormalizerInput);
    free(hostNormalizerOutput);
    free(hostBatchNormalizedQuery);
    free(hostOutputSums);
    free(hostOutputSquaredSums);
    free(hostReference);
    free(hostDTW);
  return 0;
}
