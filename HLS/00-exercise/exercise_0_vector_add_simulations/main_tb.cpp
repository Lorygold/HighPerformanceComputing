#include <stdlib.h>
#include <stdio.h>

#include "vadd.h"

int main () {

	int n = max_elem;

	int * a = (int *) malloc(n*sizeof(int));
	int * b = (int *) malloc(n*sizeof(int));
	int * c = (int *) malloc(n*sizeof(int));

	for (int i = 0; i < n; i++) {
		a[i] =   i;
		b[i] = 2*i;
	}

	sum (a, b, c, n);

	FILE *fp = fopen("ground_truth.txt", "r");

	for (int i = 0; i < n; i++) {

		int buff;

		fscanf(fp, "%d", &buff);

		if (buff != c[i]) {
			return 1;
		}
	}

	fclose(fp);

	free(a);
	free(b);
	free(c);

	return 0;
}
