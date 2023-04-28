#ifndef __PPC_Option__h_
#define __PPC_Option__h_

#include "Path.cuh"
#include "Option.cuh"
#include <cmath>

using namespace std;


class PPC_Option: public Option {
  private:
    double _strike;
    double _sigma;
    double _deltat;
    double _notional_value;
    int* _fixing_data_vector;
    double _upper_barrier_limit;
    double _lower_barrier_limit;
    int _number_of_fixing_dates;
  public:
    __device__ __host__ PPC_Option(double strike, double sigma, double delta_t, double notional_value, double lower_barrier_lim, double upper_barrier_lim, int number_of_fixing_dates, int* fixing_data_vector);
    __device__ __host__ ~PPC_Option();
    __device__ __host__ double PayOff(Path* path, int threadIdx);
};


PPC_Option::PPC_Option(double strike, double sigma, double delta_t, double notional_value,
  double lower_barrier_lim, double upper_barrier_lim, int number_of_fixing_dates, int* fixing_data_vector){
  _strike = strike;
  _sigma = sigma;
  _deltat = delta_t;
  _notional_value = notional_value;
  _fixing_data_vector = fixing_data_vector;
  _number_of_fixing_dates = number_of_fixing_dates;
  _lower_barrier_limit = lower_barrier_lim;
  _upper_barrier_limit = upper_barrier_lim;
}

PPC_Option::~PPC_Option(){}

double PPC_Option::PayOff(Path* path, int threadIdx){

  unsigned int number_of_intervals = path->get_number_of_intervals();
  unsigned int m = number_of_intervals*threadIdx;
  double* path_vector = path->CreatingPath();
  double cost = 1./sqrt(_deltat);
  double appo, p_i=0;

  int j,k;
  for(int i=0; i<_number_of_fixing_dates-1; i++){ //Attenzione _number_of_fixing_dates non può essere nullo!
    j=_fixing_data_vector[i];
    k=_fixing_data_vector[i+1];
    appo=cost*log(path_vector[m+k]/path_vector[m+j]);

    if(appo > _lower_barrier_limit && appo < _upper_barrier_limit*_sigma){
      p_i += 1.;
    }
  }
  p_i /= double(_number_of_fixing_dates);

  return _notional_value*max( 0. , p_i - _strike );
}

#endif
