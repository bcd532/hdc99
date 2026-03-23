#include "hdc_binary.h"
#include <stdint.h>
#include <stdlib.h>

static int flip_order_binary[HDC_MAX_DIMENSION];

static int flip_order_initialized_binary = 0;

void shuffle_binary(int *array, int shuffle_amount ){
    for (int i = shuffle_amount -1; i >0; i--){
        int j = rand() % (i+1);
        int temp_array = array[i];
        array[i] = array[j];
        array[j] = temp_array;
    }
}

int classify_binary(struct hdc_binary_classifier *clf, uint64_t *vector){
    for (int i = 0; i < clf->class_count; i++){
    zero_vector_binary(clf->vector[i], clf->dimension);

    for (int b = 0; b < clf->dimension; b++){
        if (clf->accum[i][b] > 0){
            clf->vector[i][b/64] |= (uint64_t)1 << (b%64);}
        }
    }
    int best_distance = clf->dimension +1;
    int bestclass = -1;
    int distance;
    for (int c = 0; c < clf->class_count; c++){
        similize_binary(&distance, vector, clf->vector[c], clf->dimension);
        if (distance < best_distance){
            bestclass = c;
            best_distance = distance;
        }
    }
    return bestclass;
}

void train_binary(struct hdc_binary_classifier *clf, uint64_t *vector, int classnum){
    if (classnum >= HDC_MAX_CLASSES || classnum < 0) return;

    for (int b = 0; b < clf->dimension; b++){
        if ((vector[b/64] >> (b%64)) & 1){
            clf->accum[classnum][b] += 1;
        }else{clf->accum[classnum][b] -= 1;}
    }
    if (classnum >= clf->class_count) clf->class_count = classnum +1;
}

void hdc_classifier_init_binary(struct hdc_binary_classifier *clf, int dimension){
clf->class_count = 0;
clf->dimension = dimension;
for (int i = 0; i < HDC_MAX_CLASSES; i++){
    zero_vector_binary(clf->vector[i], clf->dimension);
    for (int b = 0; b < clf->dimension; b++)clf->accum[i][b] =0;

}
}

void hdc_init_binary(unsigned int seed){
    srand(seed);
    for (int i = 0; i < HDC_MAX_DIMENSION; i++){
        flip_order_binary[i] = i;
    }
    shuffle_binary(flip_order_binary, HDC_MAX_DIMENSION);
    flip_order_initialized_binary =1;
}

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

void zero_vector_binary(uint64_t *vector, int dimension){
    for (int i = 0; i < HDC_BITS_TO_WORDS(dimension); i++){
        vector[i] = 0;
    }
}



void copy_vector_binary(uint64_t *dest, uint64_t *src, int dimension){

    for (int i = 0; i < HDC_BITS_TO_WORDS(dimension); i++){
        dest[i] = src[i];
    }
}

void level_encode_binary(float  value, uint64_t *result, int dimension){
    zero_vector_binary(result, dimension);
    if (value < 0){value =0;}
    else if (value > 1){value = 1;}
    int flip_amt = value * dimension;


    for (int i = 0; i < flip_amt; i++){
        int bit = flip_order_binary[i] % dimension;
        result[bit/64] |= (uint64_t)1 << (bit % 64);
        }


}

void id_level_encode_binary(float *values, uint64_t **ids, int channel_amount, uint64_t *result_vector, int dimension){
    if(channel_amount <= 0) return;
    uint64_t temp_vector[HDC_BITS_TO_WORDS(dimension)];
    uint64_t temp_vector2[HDC_BITS_TO_WORDS(dimension)];
    uint64_t bound_vecs[channel_amount][HDC_BITS_TO_WORDS(dimension)];
    uint64_t *ptrs[channel_amount];
    zero_vector_binary(result_vector, dimension);
    for (int i = 0; i < channel_amount; i++){
        level_encode_binary(values[i], temp_vector, dimension);
        bind_binary(temp_vector2,temp_vector,ids[i], dimension);
        copy_vector_binary(bound_vecs[i], temp_vector2, dimension);
        ptrs[i] = bound_vecs[i];
    }
    bundle_binary(result_vector, ptrs, channel_amount,dimension);
}

void permute_binary(uint64_t *vector, int shift_amount, uint64_t *result_vector, int dimension){
    for (int w = 0; w < HDC_BITS_TO_WORDS(dimension); w++){
        result_vector[w] = 0;
    }
    for (int b = 0; b < dimension; b++){
        int dest = ((b + shift_amount) % dimension + dimension) % dimension;
        if ((vector[b/64] >> (b % 64)) & 1){
            result_vector[dest / 64] |= (uint64_t)1 << (dest % 64);
            }
        }
    }
