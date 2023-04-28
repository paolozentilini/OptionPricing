#ifndef __WienerProcess__h_
#define __WienerProcess__h_

#include "StochasticProcess.cuh"
#include <cmath>

using namespace std;


class WienerProcess: public StochasticProcess{
  private:
    double _r, _sigma;
  protected:
    double _AssetPrice0, _AssetPricet;
  public:
    //Costruttore e distruttore:
    __device__ __host__ WienerProcess(double asset_price0, double risk_free_rate, double sigma);
    __device__ __host__ ~WienerProcess();
    //Metodo Eulero per la valutazione del salto singolo per il prezzo del sottostante:
    __device__ __host__ double MakeAStep(double rnd, double delta_t);
    //Metodi di modifica e accesso del prezzo dell'asset:
    __device__ __host__ void Set_AssetPrice(){ _AssetPricet=_AssetPrice0; }
};



WienerProcess::WienerProcess(double asset_price0, double risk_free_rate, double sigma){
  _r = risk_free_rate;
  _sigma = sigma;
  _AssetPrice0 = asset_price0;
  _AssetPricet = asset_price0;
}

WienerProcess::~WienerProcess(){}


double WienerProcess::MakeAStep(double rnd, double delta_t){
  _AssetPricet = _AssetPricet*( 1 + _r*(delta_t) + _sigma * sqrt(delta_t) * rnd );
  return _AssetPricet;
}

#endif
