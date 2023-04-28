#ifndef __PlainVanilla__h_
#define __PlainVanilla__h_

#include "Path.cuh"
#include "Option.cuh"
#include <cmath>

using namespace std;

class PlainVanilla: public Option {
  private:
    double _strike;
    char _option_type;
  public:
    __device__ __host__ PlainVanilla(double strike,char option_type);
    __device__ __host__ ~PlainVanilla();
    __device__ __host__ double PayOff(Path* path,int threadIdx);
};


PlainVanilla::PlainVanilla(double strike,char option_type){
  _strike=strike;
  _option_type=option_type;
}

PlainVanilla::~PlainVanilla(){}

double PlainVanilla::PayOff(Path* path,int threadIdx){
  unsigned int number_of_intervals = path->get_number_of_intervals();
  double* path_vector = path->CreatingPath();
  unsigned int k = number_of_intervals*threadIdx;
  if(_option_type == 'C'){
    return max(path_vector[k+number_of_intervals-1] -_strike, 0.);
  }else{
    return max(_strike - path_vector[k+number_of_intervals-1], 0.);
  }
}

#endif
