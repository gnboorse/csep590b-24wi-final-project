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

int main(int argc, char **argv) {
    srand(time(NULL));
    // wbArg_t args;
    float *hostNormalizerInput; // the reference/query string we want to normalize
    float *hostBatchNormalizerInput;
    float *hostNormalizerOutput; // the result of normalizing the string
    float *hostBatchNormalizerOutput;
    float *hostOutputSums; // summation of all items in input
    float *hostOutputSquaredSums; // sum of all squares in input (used for variance calculation)

    float *deviceNormalizerInput;
    float *deviceNormalizerOutput;
    float *deviceOutputSums;
    float *deviceOutputSquaredSums;

    // TODO: maybe use args to control size of randomly generated file?
    // args = wbArg_read(argc, argv);

    // generate a randomized query file
    wbTime_start(Generic, "Generating random query");
    wbGenerateParams_t params;
    params.raw.rows   = QUERY_BATCH;
    params.raw.cols   = QUERY_LENGTH;
    params.raw.minVal = MIN_RANDOM_VAL;
    params.raw.maxVal = MAX_RANDOM_VAL;
    params.raw.type   = wbType_float;
    
    // hardcode the timestamp here to retrieve a previously-generated file
    std::time_t current_timestamp = std::time(nullptr);

    char *inputFileName = wbPath_join(wbDirectory_current(), "query-string", "query.raw");
    char *outputFileName = wbPath_join(wbDirectory_current(), "query-string", "normalized_query.raw");
    wbDataset_generate(inputFileName, wbExportKind_raw, params);
    wbTime_stop(Generic, "Generating random query");
    wbLog(TRACE, "Creaated ", QUERY_BATCH ,"random querys of length ", QUERY_LENGTH, " at test-normalizer", current_timestamp, "query.raw");


    // import the file just generated
    wbTime_start(Generic, "Importing data and creating memory on host");
    int inputLength = QUERY_LENGTH;
    int queryLength, numQuerys;
    hostBatchNormalizerInput = (float *)wbImport(inputFileName, &numQuerys, &queryLength);
    wbLog(TRACE, "Number of Queries : ", numQuerys);
    wbLog(TRACE, "Size of each Querie : ", queryLength);
   

     
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
    hostNormalizerInput = (float *)malloc(inputSizeBytes);
    hostBatchNormalizerOutput = (float *)malloc(batchSizeBytes);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbTime_start(GPU, "Allocating GPU memory.");
    // allocate device memory
    wbCheck(hipMalloc(&deviceNormalizerInput, inputSizeBytes));
    wbCheck(hipMalloc(&deviceNormalizerOutput, inputSizeBytes));
    wbCheck(hipMalloc(&deviceOutputSums, blockSumsSizeBytes));
    wbCheck(hipMalloc(&deviceOutputSquaredSums, blockSumsSizeBytes));
    wbTime_stop(GPU, "Allocating GPU memory.");

    //--Start processing each query
    for(int currentQuery = 0; currentQuery < numQuerys; currentQuery++) {
        for(int i = 0; i < queryLength; i++) {
            hostNormalizerInput[i] = hostBatchNormalizerInput[(currentQuery * queryLength) +i];
        }
        wbTime_start(GPU, "Copying input memory to the GPU.");
        // copy input to GPU
        wbCheck(hipMemcpy(deviceNormalizerInput, hostNormalizerInput, inputSizeBytes, hipMemcpyHostToDevice));
        wbTime_stop(GPU, "Copying input memory to the GPU.");

        // initialize grid/block dimensions for the kernel that calculates sums
        dim3 gridDimensions_sums(numBlockSums, 1, 1);
        dim3 blockDimensions_sums(BLOCK_SIZE, 1, 1);

        wbTime_start(Compute, "Performing HIP computation");
        // calculate all sums and squared sums
        hipLaunchKernelGGL(
            calculate_mean_and_variance, gridDimensions_sums, blockDimensions_sums, 0, 0, 
            deviceNormalizerInput, deviceOutputSums, deviceOutputSquaredSums, inputLength
        );

        hipDeviceSynchronize();
        wbTime_stop(Compute, "Performing HIP computation");
        
        // copy output sums and squared sums back to host
        wbTime_start(Copy, "Copying output memory to the CPU");
        // Copy the GPU memory back to the CPU here
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

        wbLog(TRACE, "mu: ", mean, ", sigma: ", 1 / variance);

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
            deviceNormalizerInput, deviceNormalizerOutput, inputLength
        );

        hipDeviceSynchronize();
        wbTime_stop(Compute, "Performing HIP computation");

        wbTime_start(Copy, "Copying data from the GPU");
        // Copy the device normalized array back to the host 
        wbCheck(hipMemcpy(hostNormalizerOutput, deviceNormalizerOutput, inputSizeBytes, hipMemcpyDeviceToHost));
        wbTime_stop(Copy, "Copying data from the GPU");

        for(int i = 0; i < queryLength; i++) {
            hostBatchNormalizerOutput[(currentQuery * queryLength) + i] = hostNormalizerOutput[i];
        }
    }
    // write output to raw file
    wbExport(outputFileName, (wbReal_t *)hostBatchNormalizerOutput, numQuerys, queryLength);

    wbLog(TRACE, "normalized ", queryLength, " values");

    // Free device memory
    hipFree(deviceNormalizerInput);
    hipFree(deviceNormalizerOutput);
    hipFree(deviceOutputSums);
    hipFree(deviceOutputSquaredSums);

    // Free host memory
    free(hostNormalizerInput);
    free(hostBatchNormalizerInput);
    free(hostNormalizerOutput);
    free(hostBatchNormalizerOutput);
    free(hostOutputSums);
    free(hostOutputSquaredSums);


  return 0;
}
