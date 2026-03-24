#include "hdc_binary.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static int flip_order_binary[HDC_MAX_DIMENSION];

static int flip_order_initialized_binary = 0;

/* dimension check for binary - returns 1 if bad, 0 if ok */
static int check_dimension_binary(int dimension){
    if (dimension <= 0){
        printf("WARNING: dimension (%d) must be greater than 0\n", dimension);
        return 1;}
    if (dimension > HDC_MAX_DIMENSION){
        printf("WARNING: dimension (%d) is above max (%d)\n", dimension, HDC_MAX_DIMENSION);
        return 1;}
    if (dimension % 64 != 0){
        printf("WARNING: binary dimension (%d) must be a multiple of 64\n", dimension);
        return 1;}
    return 0;
}

/* NULL check for binary - returns 1 if any pointer is NULL */
static int check_null_binary(const void **ptrs, int count, const char *func_name){
    for (int i = 0; i < count; i++){
        if(ptrs[i] == NULL){
            printf("WARNING: NULL pointer at arg %d in %s\n", i, func_name);
            return 1;
        }
    }
    return 0;
}

void shuffle_binary(int *array, int shuffle_amount ){
    for (int i = shuffle_amount -1; i >0; i--){
        int j = rand() % (i+1);
        int temp_array = array[i];
        array[i] = array[j];
        array[j] = temp_array;
    }
}

void build_prototypes_binary(struct hdc_binary_classifier *clf){
    const void *args[] = {clf};
    if (check_null_binary(args, 1, "build_prototypes_binary"))
    for (int i = 0; i < clf->class_count; i++){
    for (int b = 0; b < clf->dimension; b++){
        if (clf->accum[i][b] > 0){
            clf->vector[i][b/64] |= (uint64_t)1 << (b%64);}
        }
    }
}

int classify_binary(struct hdc_binary_classifier *clf, uint64_t *vector){
    const void *args[] = {clf, vector};
    if (check_null_binary(args, 2, "classify_binary")) return -1;

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
    const void *args[] = {clf, vector};
    if (check_null_binary(args, 2, "train_binary")) return;
    if (classnum >= HDC_MAX_CLASSES || classnum < 0) return;

    for (int b = 0; b < clf->dimension; b++){
        if ((vector[b/64] >> (b%64)) & 1){
            clf->accum[classnum][b] += 1;
        }else{clf->accum[classnum][b] -= 1;}
    }
    if (classnum >= clf->class_count) clf->class_count = classnum +1;
}

void hdc_classifier_init_binary(struct hdc_binary_classifier *clf, int dimension){
    const void *args[] = {clf};
    if (check_null_binary(args, 1, "hdc_classifier_init_binary")) return;
    if (check_dimension_binary(dimension)) return;
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
    const void *args[] = {vector};
    if (check_null_binary(args, 1, "random_binary")) return;
    if (check_dimension_binary(dimension)) return;
    int words = HDC_BITS_TO_WORDS(dimension);
    for (int i = 0; i < words; i++){
        vector[i] = 0;
        for (int h = 0; h < 64; h++){
            if(rand() % 2){
                vector[i] |= (uint64_t)1 << h;
            }
        }
    }
    int leftover = dimension % 64;
    if (leftover > 0){
        vector[words - 1] &= ((uint64_t)1 << leftover) - 1;
    }
}

void bind_binary(uint64_t *result_vector, uint64_t *vectora, uint64_t *vectorb, int dimension){
    const void *args[] = {result_vector, vectora, vectorb};
    if (check_null_binary(args, 3, "bind_binary")) return;
    if (check_dimension_binary(dimension)) return;
    int words = HDC_BITS_TO_WORDS(dimension);
    for (int i = 0; i < words; i++){
        result_vector[i] = vectora[i] ^ vectorb[i];
    }
}

void similize_binary(int *similarity_int, uint64_t *vectora, uint64_t *vectorb, int dimension){
    const void *args[] = {similarity_int, vectora, vectorb};
    if (check_null_binary(args, 3, "similize_binary")) return;
    if (check_dimension_binary(dimension)) return;
    int words = HDC_BITS_TO_WORDS(dimension);
    int distance = 0;
    for (int i = 0; i < words; i++){
        distance += __builtin_popcountll(vectora[i] ^ vectorb[i]);
    }
    *similarity_int = distance;
}

void bundle_binary(uint64_t *result_vector, uint64_t **vectors,int count, int dimension){
    const void *args[] = {result_vector, vectors};
    if (check_null_binary(args, 2, "bundle_binary")) return;
    if (check_dimension_binary(dimension)) return;
    if (count <= 0) return;
    for (int w = 0; w < HDC_BITS_TO_WORDS(dimension); w++){
        result_vector[w] = 0;
    }
    for (int b = 0; b < dimension; b++){
        int tally = 0;
        int word = b / 64;
        int bit = b % 64;

        for (int v = 0; v < count; v++){
            if (vectors[v] == NULL) continue;
            if (vectors[v][word] & ((uint64_t)1 << bit))tally++;
        }
        if (tally > count/2){
            result_vector[word] |= (uint64_t)1 <<bit;
        }
    }
}

void zero_vector_binary(uint64_t *vector, int dimension){
    const void *args[] = {vector};
    if (check_null_binary(args, 1, "zero_vector_binary")) return;
    for (int i = 0; i < HDC_BITS_TO_WORDS(dimension); i++){
        vector[i] = 0;
    }
}

void copy_vector_binary(uint64_t *dest, uint64_t *src, int dimension){
    const void *args[] = {dest, src};
    if (check_null_binary(args, 2, "copy_vector_binary")) return;
    for (int i = 0; i < HDC_BITS_TO_WORDS(dimension); i++){
        dest[i] = src[i];
    }
}

void level_encode_binary(float  value, uint64_t *result, int dimension){
    const void *args[] = {result};
    if (check_null_binary(args, 1, "level_encode_binary")) return;
    if (check_dimension_binary(dimension)) return;
    if(!flip_order_initialized_binary){
        printf("WARNING: flip_order_binary not initialized! Call hdc_init_binary() first.\n"); return;
    }
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
    const void *args[] = {values, ids, result_vector};
    if (check_null_binary(args, 3, "id_level_encode_binary")) return;
    if (check_dimension_binary(dimension)) return;
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
    const void *args[] = {vector, result_vector};
    if (check_null_binary(args, 2, "permute_binary")) return;
    if (check_dimension_binary(dimension)) return;
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
