#include <hip/hip_runtime.h>
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

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
  wbArg_t args;
  float *hostQuery; // The Query vector
  float *hostReference; // The Reference vector
  float *hostDTW; // The output DTW matrix
  float *deviceQuery;
  float *deviceReference;
  float *deviceDTW;
  int queryLength;    // number of items in the Query
  int referenceLength;    // number of items in the Reference
  int DTWRows;    // number of rows in the DTW 
  int DTWColumns; // number of columns in the DTW

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostQuery = (float *)wbImport(wbArg_getInputFile(args, 0), &queryLength);
  hostReference = (float *)wbImport(wbArg_getInputFile(args, 1), &referenceLength);
  DTWRows = queryLength;
  DTWColumns = referenceLength;
  //@@ Allocate the hostDTW matrix
  hostDTW = (float *)malloc(DTWRows * DTWColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The length of A is ", queryLength);
  wbLog(TRACE, "The length of B is ", referenceLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  hipMalloc(&deviceQuery, queryLength * sizeof(float));
  hipMalloc(&deviceReference, referenceLength * sizeof(float));
  hipMalloc(&deviceDTW, DTWRows * DTWColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  hipMemcpy(deviceQuery, hostQuery, queryLength * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(deviceReference, hostReference, referenceLength * sizeof(float), hipMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  unsigned int blockSize = 32;
  dim3 threads(blockSize,blockSize,1);
  dim3 blocks((DTWColumns+blockSize-1)/blockSize, (DTWRows+blockSize-1)/blockSize, 1);

  wbTime_start(Compute, "Performing HIP computation");
  //@@ Launch the GPU Kernel here
  hipLaunchKernelGGL(dtw_matrix, blocks, threads, 0, 0, deviceQuery, deviceReference, deviceDTW, queryLength,
    referenceLength, DTWRows, DTWColumns);
  
  hipDeviceSynchronize();
  wbTime_stop(Compute, "Performing HIP computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  hipMemcpy(hostDTW, deviceDTW, DTWRows * DTWColumns * sizeof(float), hipMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  hipFree(deviceQuery);
  hipFree(deviceReference);
  hipFree(deviceDTW);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostDTW, DTWRows, DTWColumns);

  free(hostQuery);
  free(hostReference);
  free(hostDTW);

  return 0;
}
