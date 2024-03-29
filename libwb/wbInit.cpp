
#include "wb.h"

#define MB (1 << 20)
#ifndef WB_DEFAULT_HEAP_SIZE
#define WB_DEFAULT_HEAP_SIZE (1024 * MB)
#endif /* WB_DEFAULT_HEAP_SIZE */

static bool _initializedQ = wbFalse;

#ifndef WB_USE_WINDOWS
//__attribute__((__constructor__))
#endif /* WB_USE_WINDOWS */
void wb_init(int *
#ifdef WB_USE_MPI
                 argc
#endif /* WB_USE_MPI */
             ,
             char ***
#ifdef WB_USE_MPI
                 argv
#endif /* WB_USE_MPI */
             ) {
  if (_initializedQ == wbTrue) {
    return;
  }
#ifdef WB_USE_MPI
  wbMPI_Init(argc, argv);
#endif /* WB_USE_MPI */

  _envSessionId();
#ifdef WB_USE_HIP
  CUresult err = cuInit(0);

/* Select a random GPU */

#ifdef WB_USE_MPI
  if (rankCount() > 1) {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    srand(time(NULL));
    hipSetDevice(wbMPI_getRank() % deviceCount);
  } else {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);

    srand(time(NULL));
    hipSetDevice(rand() % deviceCount);
  }
#else
  {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);

    srand(time(NULL));
    hipSetDevice(rand() % deviceCount);
  }
#endif /* WB_USE_MPI */

  hipDeviceSetLimit(hipLimitPrintfFifoSize, 1 * MB);
  hipDeviceSetLimit(hipLimitMallocHeapSize, WB_DEFAULT_HEAP_SIZE);

  hipDeviceSynchronize();

#endif /* WB_USE_HIP */

#ifdef WB_USE_WINDOWS
  QueryPerformanceFrequency((LARGE_INTEGER *)&_hrtime_frequency);
#endif /* _MSC_VER */

  _hrtime();

  _timer        = wbTimer_new();
  _logger       = wbLogger_new();
  _initializedQ = wbTrue;

  wbFile_init();

  solutionJSON = nullptr;

#ifdef WB_USE_MPI
  atexit(wbMPI_Exit);
#else  /* WB_USE_MPI */
  atexit(wb_atExit);
#endif /* WB_USE_MPI */
}
