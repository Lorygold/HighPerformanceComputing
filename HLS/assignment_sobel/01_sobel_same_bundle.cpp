#include <math.h>
#include "sobel.h"
#define HEIGHT 512
#define WIDTH 512

/*
Utilizzare anche un unico bundle per risparmiare risorse, in quanto ogni bundle 
ha 1 canale per la scrittura e 1 canale per la lettura e noi abbiamo solamente l'input da leggere 
e l'output da scrivere
*/

void sobel(uint8_t *__restrict__ out, uint8_t *__restrict__ in, int width, int height)
{
    // interfaces for arrays
    #pragma HLS INTERFACE m_axi port=in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=mem
    // interfaces for scalar var
    #pragma HLS INTERFACE s_axilite port=width bundle=params
    #pragma HLS INTERFACE s_axilite port=height bundle=params
    #pragma HLS INTERFACE s_axilite port=return bundle=params

    int sobelFilter[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    for (int y = 1; y < height - 1; y++)
    {
        #pragma HLS LOOP_TRIPCOUNT min=HEIGHT max=HEIGHT
        for (int x = 1; x < width - 1; x++)
        {
            #pragma HLS LOOP_TRIPCOUNT min=WIDTH max=WIDTH
            int dx = 0, dy = 0;
            for (int k = 0; k < 3; k++)
            {
                for (int z = 0; z < 3; z++)
                {
                    dx += sobelFilter[k][z] * in[(y + k - 1) * width + x + z - 1];
                    dy += sobelFilter[z][k] * in[(y + k - 1) * width + x + z - 1];
                }
            }
            out[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
        }
    }
}
