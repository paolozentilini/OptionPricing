#ifndef __SimulationMonteCarlo__h__
#define __SimulationMonteCarlo__h__

#include "Simulation.cuh"
#include "Path.cuh"
#include "Option.cuh"

using namespace std;

class SimulationMonteCarlo: public Simulation{
  private:
    unsigned int _Nscenarios;
    Path *_path;
    Option *_option;
    int _threadIdx;
  public:
    __device__ __host__ SimulationMonteCarlo(int Nscenarios, int threadIdx, Path *path, Option *option);
    __device__ __host__ ~SimulationMonteCarlo();
    __device__ __host__ void OptionPricing(double* mean_value, double* squared_mean_value);
};


SimulationMonteCarlo::SimulationMonteCarlo(int Nscenarios, int threadIdx, Path* path, Option* option){
  _Nscenarios=Nscenarios;
  _path=path;
  _option=option;
  _threadIdx=threadIdx;
}

SimulationMonteCarlo::~SimulationMonteCarlo(){}

void SimulationMonteCarlo::OptionPricing(double* mean_value, double* squared_mean_value){

  double data=0, datasquared=0, appo;
  for(unsigned int i=0; i<_Nscenarios; i++){
    appo = _option->PayOff(_path,_threadIdx);
    data += appo;
    datasquared += appo * appo;
  }
  mean_value[_threadIdx] = data;
  squared_mean_value[_threadIdx] = datasquared;
}


#endif
