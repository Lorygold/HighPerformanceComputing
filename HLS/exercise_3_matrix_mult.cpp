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

    for (int i = 0; i < dim; i++){
		#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
        for (int j = 0; j < dim; j++){
			#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
            for (int k = 0; k < dim; k++){
                //TODO: insert pipeline directive
				#pragma HLS PIPELINE
				#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
                out[i * dim + j] += in1[i * dim + k] * in2[k * dim  + j];
            }
        }
    }
}
