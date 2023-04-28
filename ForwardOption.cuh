#ifndef __Forward_Option__h_
#define __Forward_Option__h_

#include "Path.cuh"
#include "Option.cuh"
#include <cmath>

using namespace std;

class ForwardOption: public Option{
  private:
  double _strike;
  public:
    __device__ __host__ ForwardOption(double strike);
    __device__ __host__ ~ForwardOption();
    __device__ __host__ double PayOff(Path* path,int threadIdx);
};


ForwardOption::ForwardOption(double strike){
  _strike = strike;
}

ForwardOption::~ForwardOption(){}

double ForwardOption::PayOff(Path* path, int threadIdx){
  unsigned int number_of_intervals = path->get_number_of_intervals();
  double* path_vector = path->CreatingPath();
  unsigned int k = number_of_intervals*threadIdx;
  return path_vector[k+number_of_intervals-1];
}

#endif
