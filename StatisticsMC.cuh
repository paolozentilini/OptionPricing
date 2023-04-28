#ifndef __StatisticsMC__h__
#define __StatisticsMC__h__

#include "Statistics.cuh"
#include "Data.cuh"

using namespace std;

class StatisticsMC: public Statistics{
  private:
    unsigned int _N_elements; //Numero di elementi dei vettori uscenti dalla gpu
    unsigned int _Nscenarios; //Numero totale di scenari sui quali calcolare media ed errore
    double _mean, _mean_squared;
    double _risk_free_rate;
    double _total_time;
  public:
    __host__ StatisticsMC(Input_Data* data);
    __host__ ~StatisticsMC();
    __host__ double mean(double* mean_vector);
    __host__ double error(double* mean_squared_vector);
};


StatisticsMC::StatisticsMC(Input_Data* data){
   _N_elements = data->N;
   _Nscenarios = data->Nscenarios;
   _mean =0;
   _mean_squared=0;
   _risk_free_rate=data->risk_free_rate;
   _total_time=data->total_time;
}

StatisticsMC::~StatisticsMC(){}

double StatisticsMC::mean(double* mean_vector){
  for(int i=0; i<_N_elements; i++){
    _mean += mean_vector[i];
  }
  _mean /= _Nscenarios;
  return _mean ;//exp(-_risk_free_rate*_total_time)*
}

double StatisticsMC::error(double* mean_squared_vector){
  for(int i=0; i<_N_elements; i++) {
    _mean_squared += mean_squared_vector[i];
  }
  _mean_squared /= _Nscenarios;
  double error = (1/sqrt(double(_Nscenarios)))*sqrt( _mean_squared - _mean*_mean);
  return error;
}

#endif
