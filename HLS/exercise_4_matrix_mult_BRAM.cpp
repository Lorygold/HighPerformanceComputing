#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 64

const unsigned int max_size = MAX_SIZE;

void mmult( int *in1,
            int *in2,
            int *out,
            int dim
              )
{
#pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=in1_mem
#pragma HLS INTERFACE m_axi port=in2 offset=slave bundle=in2_mem
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=out_mem

#pragma HLS INTERFACE s_axilite port=dim bundle=params
#pragma HLS INTERFACE s_axilite port=return bundle=params

    //TODO: create three Blocked RAM
    int locIn1[MAX_SIZE][MAX_SIZE];
    int locIn2[MAX_SIZE][MAX_SIZE];
    int locOut[MAX_SIZE][MAX_SIZE];
    //TODO: copy data from DRAM to BRAM
    memcpy(locIn1, in1, MAX_SIZE*MAX_SIZE*sizeof(int));
    memcpy(locIn2, in2, MAX_SIZE*MAX_SIZE*sizeof(int));
    // out_buffer is empty, so it's useless to copy it

    for (int i = 0; i < dim; i++){
		#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
        for (int j = 0; j < dim; j++){
			#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
            // Inizialize the output to 0
            locOut[i][j] = 0;
            for (int k = 0; k < dim; k++){
                //TODO: insert pipeline directive
                #pragma HLS PIPELINE
				#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
                locOut[i][j] += locIn1[i][k] * locIn2[k][j];
            }
        }
    }
    //TODO: copy data back from BRAM to DRAM (just the result)
    memcpy(out, locOut, MAX_SIZE*MAX_SIZE*sizeof(int));
}
