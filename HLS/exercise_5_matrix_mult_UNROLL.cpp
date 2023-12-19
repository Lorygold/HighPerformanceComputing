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
    //TODO: insert array partition directives


    for (int i = 0; i < dim; i++){
		#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
        for (int j = 0; j < dim; j++){
            //TODO: insert pipeline directive
            #pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
            // do the k iterations (the second dimention of the matrix) in parallel with the PIPELINE pragma before the for loop on k
            for (int k = 0; k < dim; k++){
                //TODO: insert loop unrolling directive
                // it's the same of putting the HLS PIPELINE before the loop
				#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
                out[i * dim + j] += in1[i * dim + k] * in2[k * dim  + j];
            }
        }
    }
    //TODO: copy data back from BRAM to DRAM
}
