#ifndef __Statistics__h__
#define __Statistics__h__

class Statistics{
  public:
    __host__ virtual double mean(double* mean_vector)=0;
    __host__ virtual double error(double* mean_squared_vector)=0;
};

#endif
