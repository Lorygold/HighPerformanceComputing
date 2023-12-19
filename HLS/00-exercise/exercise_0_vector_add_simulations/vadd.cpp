#include "vadd.h"

void sum (int *a, int *b, int *c, int n) {

#pragma HLS INTERFACE s_axilite port=n bundle=regfile
#pragma HLS_INTERFACE s_axilite port=return bundle=regfile
#pragma HLS INTERFACE m_axi port=a offset=slave depth=max_elem bundle=a_mem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=max_elem bundle=bc_mem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=max_elem bundle=bc_mem

	for (int i = 0; i < n; i++) {
	#pragma HLS LOOP_TRIPCOUNT min=max_elem max=max_elem
		c[i] = a[i] + b[i];
	}
}
