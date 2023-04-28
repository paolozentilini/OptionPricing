#ifndef __CombinedGenerator__h_
#define __CombinedGenerator__h_

#include "RandomGenerator.cuh"
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

//Classe concreta figlia di RandomGenerator: Combined Generator
class CombinedGenerator : public RandomGenerator{
    private:
      unsigned int _vectorseeds[4];
      unsigned int _a, _b, _M;  //Parametri di Lcg e Taussworthe
    public:
      __device__ __host__ CombinedGenerator(unsigned int vectorseeds[4]);
      __device__ __host__ ~CombinedGenerator();

      __device__ __host__ unsigned int Lcg(int index);
      __device__ __host__ unsigned int Taussworthe(unsigned int k1, unsigned int k2, unsigned int k3, unsigned int m, int index);

      __device__ __host__ double get_uniform();
      __device__ __host__ double get_uniform(double min, double max);
      __device__ __host__ double get_gaussian(double mean, double sigma);
      __device__ __host__ double get_bimodal();
};

//////////////////////////////////IMPLEMENTAZIONE DEI METODI////////////////////////////////////////////////////////////

CombinedGenerator::CombinedGenerator(unsigned int vectorseeds[4]){
  _vectorseeds[0] = vectorseeds[0];
  _vectorseeds[1] = vectorseeds[1];
  _vectorseeds[2] = vectorseeds[2];
  _vectorseeds[3] = vectorseeds[3];
	_M = 4294967295;
	_a = 1664525;
	_b = 1013904223UL;
}

CombinedGenerator::~CombinedGenerator(){
}

unsigned int CombinedGenerator::Lcg(int index){
	_vectorseeds[index] = _a * (_vectorseeds[index] + _b) % _M;
	return _vectorseeds[index];
}

unsigned int CombinedGenerator::Taussworthe(unsigned int k1,unsigned int k2,unsigned int k3,unsigned int m, int index){
	unsigned b = ((( _vectorseeds[index] << k1 )^_vectorseeds[index]) >> k2);
	_vectorseeds[index] = ((( _vectorseeds[index] & m ) << k3 )^b);
	return _vectorseeds[index];
}

double CombinedGenerator::get_uniform(){
	return 2.3283064365387e-10*(
		Taussworthe( 13, 19, 12, 4294967294UL, 0)
	^	Taussworthe( 2, 25, 4, 4294967288UL, 1)
	^	Taussworthe( 3, 11, 17, 4294967280UL, 2)
	^ Lcg(3));
}

double CombinedGenerator::get_uniform(double min, double max){
	double number = 2.3283064365387e-10*(
		Taussworthe( 13, 19, 12, 4294967294UL, 0)
	^	Taussworthe( 2, 25, 4, 4294967288UL, 1)
	^	Taussworthe( 3, 11, 17, 4294967280UL, 2)
	^ Lcg(3));
	return min + number * max;
}

double CombinedGenerator::get_gaussian(double mean, double sigma){
  double x;
  do{
    double s=get_uniform();
    double t=get_uniform();
    x=sqrt(-2.*log(1.-s))*cos(2.*M_PI*t);
  }while(isnan(x)==1);
	return mean + x * sigma;
}

double CombinedGenerator::get_bimodal(){

  double r;
  double s=get_uniform();
  if(s>0.5){
    r=1;
  }else{
    r=-1;
  }
  return r;
}

#endif
