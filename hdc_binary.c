#include "hdc_binary.h"
#include "hdc.h"
#include <stdint.h>
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

void bind_binary(uint64_t *result_vector, uint64_t *vectora, uint64_t *vectorb, int dimension){
    int words = HDC_BITS_TO_WORDS(dimension);
    for (int i = 0; i < words; i++){
        result_vector[i] = vectora[i] ^ vectorb[i];
    }
}

void similize_binary(int *similarity_int, uint64_t *vectora, uint64_t *vectorb, int dimension){
    int words = HDC_BITS_TO_WORDS(dimension);
    int distance = 0;
    for (int i = 0; i < words; i++){
        distance += __builtin_popcountll(vectora[i] ^ vectorb[i]);
    }
    *similarity_int = distance;
}

void bundle_binary(uint64_t *result_vector, uint64_t **vectors,int count, int dimension){
    for (int w = 0; w < HDC_BITS_TO_WORDS(dimension); w++){
        result_vector[w] = 0;
    }
    for (int b = 0; b < dimension; b++){
        int tally = 0;
        int word = b / 64;
        int bit = b % 64;

        for (int v = 0; v < count; v++){
            if (vectors[v][word] & ((uint64_t)1 << bit))tally++;
        }
        if (tally > count/2){
            result_vector[word] |= (uint64_t)1 <<bit;
        }
    }
}

void permute_binary(uint64_t *vector, int shift_amount, uint64_t *result_vector, int dimension){
    for (int w = 0; w < HDC_BITS_TO_WORDS(dimension); w++){
        result_vector[w] = 0;
    }
    for (int b = 0; b < dimension; b++){
        int dest = (b + shift_amount) % dimension;
        if ((vector[b/64] >> (b % 64)) & 1){
            result_vector[dest / 64] |= (uint64_t)1 << (dest % 64);
            }
        }
    }
