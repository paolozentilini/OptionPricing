#ifndef __Simulation__h__
#define __Simulation__h__

class Simulation{
  public:
    __device__ __host__ virtual void OptionPricing(double* mean_value, double* squared_mean_value)=0;
};

#endif
