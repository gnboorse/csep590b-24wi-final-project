
#include "wb.h"
#ifdef WB_USE_HIP

int _hipMemoryListIdx = 0;

size_t _hipMallocSize = 0;

wbHIPMemory_t _hipMemoryList[_hipMemoryListSize];

char *wbRandom_list(size_t sz) {
  size_t ii;
  char *rands = wbNewArray(char, sz);
  int *irands = (int *)rands;
  for (ii = 0; ii < sz / sizeof(int); ii++) {
    irands[ii] = rand();
  }
  while (ii < sz) {
    rands[ii] = (char)(rand() % 255);
    ii++;
  }
  return rands;
}

#endif /* WB_USE_HIP */
