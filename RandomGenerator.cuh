#ifndef __RandomGenerator__h_
#define __RandomGenerator__h_

//Classe astratta di generatori di numeri random distribuiti gaussianamente e uniformemente
class RandomGenerator{
    public:
        __device__ __host__ virtual double get_uniform()=0; //Restituisce un numero casuale uniformemente distribuito tra 0 e 1
        __device__ __host__ virtual double get_uniform(double min, double max)=0; //Restituisce un numero casuale uniformemente distribuito tra min e max
        __device__ __host__ virtual double get_gaussian(double mean, double sigma)=0; //Restituisce un numero gaussiano con varianza sigma e media mean
        __device__ __host__ virtual double get_bimodal()=0; //Restituisce una variabile stocastica bimodale
};

#endif
