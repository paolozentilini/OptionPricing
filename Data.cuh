#ifndef __Data__h__
#define __Data__h__

struct Input_Data{
    //Input_Market_Data:
    double asset_price0;
    double risk_free_rate;
    double sigma;
    //Input_Option_Data:
    double strike;
    double delta_t;
    int number_of_intervals;
    char option_type;
    double total_time;
    //Input data for Exotic Option:
    int number_of_fixing_dates;
    double upper_barrier_limit;
    double lower_barrier_limit;
    double notional_value;
    int number_of_intervals_for_fixing_data;
    //Input_Gpu_Data:
    int num_threads_per_block;
    int num_blocks;
    int N;
    //Input_MonteCarlo_Data:
    int Nscenarios_for_thread;
    int Nscenarios;
};

#endif
