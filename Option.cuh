#ifndef __Option__h_
#define __Option__h_

#include "Path.cuh"

class Option{
    public:
      __device__ __host__ virtual double PayOff(Path* path, int threadIdx)=0;
};

#endif
