
#include "PixelKernels.h"

__global__ void temporary () {
  int a;
  a = 3*4;
}

void temp_fn() {
  temporary<<<1,1>>>();
}
