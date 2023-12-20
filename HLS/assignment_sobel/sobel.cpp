#include <math.h>
#include "sobel.h"

void sobel(uint8_t *__restrict__ out, uint8_t *__restrict__ in, int width, int height)
{
    int sobelFilter[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
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
