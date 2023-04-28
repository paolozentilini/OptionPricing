#ifndef __StandardPath__h
#define __StandardPath__h

#include "RandomGenerator.cuh"
#include "StochasticProcess.cuh"
#include "Path.cuh"

using namespace std;


class StandardPath : public Path{
  protected:
    RandomGenerator *_random_generator;
    StochasticProcess *_stochastic_process;
    unsigned int _number_of_intervals;                    //N° date di fixing
    double _deltat;
    double* _trajectory;
    int _threadIdx;
  public:
    //Costruttori e distruttore:
    __device__ __host__ StandardPath(int number_of_intervals, double delta_t, int threadIdx, RandomGenerator* random_generator, StochasticProcess* stochastic_process, double* trajectory);
    __device__ __host__ ~StandardPath();
    //Funzione che restituisce il vettore dei punti successivi di una traiettoria:
    __device__ __host__ unsigned int get_number_of_intervals(){return _number_of_intervals;}
    __device__ __host__ double* CreatingPath();
};


StandardPath::StandardPath(int number_of_intervals, double delta_t, int threadIdx, RandomGenerator *random_generator, StochasticProcess *stochastic_process, double* trajectory){
  _number_of_intervals = number_of_intervals;
  _deltat = delta_t;
  _random_generator = random_generator;
  _stochastic_process = stochastic_process;
  _trajectory = trajectory;
  _threadIdx = threadIdx;
}

StandardPath::~StandardPath(){}

double* StandardPath::CreatingPath(){
  double random_number;
  _stochastic_process -> Set_AssetPrice();
  unsigned int k = _threadIdx*_number_of_intervals;
  for(unsigned int i=0; i<_number_of_intervals; i++){
    random_number = _random_generator -> get_gaussian(0,1);
    //random_number = _random_generator -> get_bimodal();
    _trajectory[i+k]= _stochastic_process -> MakeAStep(random_number,_deltat);
  }
  return _trajectory;
}

#endif
