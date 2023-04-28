#ifndef __StochasticProcess__h_
#define __StochasticProcess__h_

//Classe virtuale pura che costruisce uno step per il generico processo stocastico poi implementato nelle classi figlie concrete
class StochasticProcess{
  public:
    __device__ __host__ virtual void Set_AssetPrice()=0;
    //Make a step genera il punto successivo al tempo t+delta_t per il processo stocastico dato il tempo t e il numero random gaussiano rnd:
    __device__ __host__ virtual double MakeAStep(double rnd, double delta_t)=0;
};

#endif
