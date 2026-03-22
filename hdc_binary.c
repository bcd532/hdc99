#include "hdc_binary.h"
#include <stdlib.h>

void random_binary(uint64_t *vector, int dimension){
    int words = HDC_BITS_TO_WORDS(dimension);
    for (int i = 0; i < words; i++){
        vector[i] = 0;
        for (int h = 0; h < 64; h++){
            if(rand() % 2){
                vector[i] |= (uint64_t)1 << h;
            }
        }
    }
}
