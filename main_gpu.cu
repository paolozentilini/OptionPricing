#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "RandomGenerator.cuh"
#include "CombinedGenerator.cuh"
#include "StochasticProcess.cuh"
#include "WienerProcess.cuh"
#include "ExactWienerProcess.cuh"
#include "Path.cuh"
#include "StandardPath.cuh"
#include "Simulation.cuh"
#include "SimulationMonteCarlo.cuh"
#include "Statistics.cuh"
#include "StatisticsMC.cuh"
#include "Data.cuh"
#include "Option.cuh"
#include "PlainVanilla.cuh"
#include "ForwardOption.cuh"
#include "PPC_Option.cuh"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

__global__ void cudarandom(double *dev_mean, double *dev_mean_squared, unsigned int* dev_seeds, double* dev_trajectory, Input_Data* dev_data, int* dev_fixing_dates){

	int tid = blockIdx.x*blockDim.x + threadIdx.x;	//tid indica thread_identity.
	int j= 4*tid;
	//Costruzione dell'array a 4 semi che deve essere usato per il funzionamento del generatore di numeri random
	unsigned int seeds[4];
	for(int i=0; i<4; i++){
		seeds[i] = dev_seeds[j];
		j+=1;
	}

	//Assegnazione del valore delle variabili utili alla simulazione:
	int Nscenarios_for_thread = dev_data->Nscenarios_for_thread;
	int number_of_intervals = dev_data->number_of_intervals;
	double asset_price0 = dev_data->asset_price0;
	double risk_free_rate = dev_data->risk_free_rate;
	double sigma = dev_data->sigma;
	double strike = dev_data->strike;
	double delta_t = dev_data->delta_t;
	double lower_barrier_lim = dev_data->lower_barrier_limit;
	double upper_barrier_lim = dev_data->upper_barrier_limit;
	int number_of_fixing_dates = dev_data->number_of_fixing_dates;
	double notional_value = dev_data->notional_value;
	char option_type = dev_data->option_type;

	//Cuore dell'algoritmo: utilizzo delle classi per il "pricing" dell'opzione finanziaria.
	CombinedGenerator generator(seeds);
	ExactWienerProcess process(asset_price0, risk_free_rate, sigma);
	StandardPath path(number_of_intervals, delta_t, tid, &generator, &process, dev_trajectory);
	//ForwardOption option();
	//PlainVanilla option(strike,option_type);
	PPC_Option option(strike, sigma, delta_t, notional_value, lower_barrier_lim, upper_barrier_lim, number_of_fixing_dates,  dev_fixing_dates);
	SimulationMonteCarlo simulation(Nscenarios_for_thread, tid, &path, &option);

	simulation.OptionPricing(dev_mean, dev_mean_squared);
}

////////////////////////////////////////////////////////////////////////////////

int main (){

int f;
cout << "fixing:" << endl;
cin >> f;
cout << "Loading.." << endl;
/////////////////////////////////INPUT DATA/////////////////////////////////////

Input_Data* input_data = new Input_Data;
//Numero di blocchi, numero di threads per blocco e numero totale di threads:
input_data->num_threads_per_block = 512;
input_data->num_blocks = 50;
input_data->N = (input_data->num_blocks)*(input_data->num_threads_per_block);
//Numero di scenari per thread, numero totale di simulazioni(o scenari)
input_data->Nscenarios_for_thread = 1000;
input_data->Nscenarios = (input_data->N)*(input_data->Nscenarios_for_thread);
//Prezzo dell'asset al tempo 0, valore del rate privo di rischio e valore della volatilità:
input_data->asset_price0 = 100;
input_data->risk_free_rate = 0.15;
input_data->sigma = 0.3;
input_data->strike = 0;
//Solo per opzione esotica:
input_data->number_of_fixing_dates = f;
input_data->lower_barrier_limit = 0;
input_data->upper_barrier_limit = 3;
input_data->notional_value = 1;
//Prezzo dello strike, numero di intervalli in cui dividere ciascuno scenario e step temporale:
input_data->number_of_intervals_for_fixing_data = 1;
input_data->number_of_intervals = input_data->number_of_intervals_for_fixing_data*input_data->number_of_fixing_dates;
input_data->delta_t = 1./(input_data->number_of_intervals);
input_data->option_type = 'C';
input_data->total_time = (input_data->delta_t)*(input_data->number_of_intervals);

////////////////////////////////////////////////////////////////////////////////
int l = input_data->number_of_intervals_for_fixing_data;
int N = input_data->N;
int IntervalsSteps = input_data->number_of_intervals;
StatisticsMC stat(input_data);
//Per opzione esotica:
int number_of_fixing_dates = input_data->number_of_fixing_dates;
int* fixing_dates = new int[number_of_fixing_dates];
for(int i=0; i<number_of_fixing_dates; i++) fixing_dates[i]= i*l;
//Dichiaro i puntatori che mi servono per passare alla gpu
double *dev_mean, *dev_mean_squared, *dev_trajectory;
unsigned int* dev_seeds;
Input_Data* dev_data;
int* dev_fixing_dates;
//Dichiaro i puntatori che immagazzinano i dati provenienti da gpu
double *mean_squared = new double[N];
double *mean = new double[N];
//Dichiaro le variabili per calcolare il tempo impiegato per il calcolo della gpu
cudaEvent_t start,stop;
float elapsed=0;
cudaEventCreate(&start);
cudaEventCreate(&stop);
//Variabili utili ad immagazzinare i valori della media e dell'errore associato:
double mean_value=0;
long double error=0;
//Creo gli array di semi che verranno usati nella gpu dal generatore di numeri random
unsigned int *seeds0= new unsigned int[4];
for(int i=0; i<4; i++){ seeds0[i] = 129 + 12*i; }
RandomGenerator* rnd = new CombinedGenerator(seeds0);
unsigned int seeds[4*N];
for(int i=0; i<4*N; i++){	seeds[i] = int(129 + rnd->get_uniform()*( 4*N - 129 )); }

//////////////////////////////////////////////////////////////////////////////

//Parte di allocazione della memoria e uso della gpu con annesso tempo di calcolo
cudaMalloc( (void **)&dev_mean, N*sizeof(double) );
cudaMalloc( (void **)&dev_mean_squared, N*sizeof(double) );
cudaMalloc( (void **)&dev_trajectory, N*IntervalsSteps*sizeof(double) );
cudaMalloc( (void **)&dev_seeds, 4*N*sizeof(unsigned int) );
cudaMalloc( (void **)&dev_data, sizeof(Input_Data) );
cudaMalloc( (void **)&dev_fixing_dates, number_of_fixing_dates*sizeof(int) );

cudaMemcpy( dev_seeds, seeds, 4*N*sizeof(unsigned int), cudaMemcpyHostToDevice);
cudaMemcpy( dev_data, input_data, sizeof(Input_Data), cudaMemcpyHostToDevice);
cudaMemcpy( dev_fixing_dates, fixing_dates, number_of_fixing_dates*sizeof(int), cudaMemcpyHostToDevice);

cudaEventRecord(start,0);

cudarandom <<< input_data->num_blocks, input_data->num_threads_per_block >>> (dev_mean, dev_mean_squared, dev_seeds, dev_trajectory, dev_data, dev_fixing_dates);		//<n° blocchi, n°threads>

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed,start,stop);

cudaMemcpy( mean, dev_mean, N*sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy( mean_squared, dev_mean_squared, N*sizeof(double), cudaMemcpyDeviceToHost);

cout << "Time taken by the gpu: " << float(elapsed)/1000. << " seconds" << endl;

//Calcolo media ed errore associato:
mean_value = stat.mean(mean);
error = stat.error(mean_squared);

//Stampo i risultati a video:
cout << setprecision(15) << "Price value:" << mean_value << endl;
cout << setprecision(15) << "Error:" << error << endl;


ofstream myfile;
myfile.open("ProbabilitàMeglio2.dat", ios_base::app); // append instead of overwrite
myfile << f-1 << "	" << setprecision(15) <<  mean_value << "	" << error << "	" << float(elapsed)/1000. << endl;
myfile.close();

////////////////////////////////////////////////////////////////////////////////

//Libero la memoria gpu ed elimino le varabili cuda-tempo:
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaFree( dev_mean);
cudaFree( dev_mean_squared);
cudaFree( dev_trajectory);
cudaFree( dev_seeds);
cudaFree( dev_data);
cudaFree( dev_fixing_dates);
//Libero la memoria su cpu:
delete[] seeds0;
delete[] mean;
delete[] mean_squared;
delete[] fixing_dates;
delete input_data;

return 0;
}
