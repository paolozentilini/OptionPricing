#ifndef __Path__h
#define __Path__h

class Path{
  public:
    __device__ __host__ virtual double* CreatingPath()=0;
    __device__ __host__ virtual unsigned int get_number_of_intervals()=0;
};

#endif
