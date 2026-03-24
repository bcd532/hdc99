#ifndef HDC_BINARY_H
#define HDC_BINARY_H

/**
 * hdc_binary.h - Hyperdimensional computing library (binary version).
 * High-dimensional vector operations for encoding and comparing data using binary vectors.
 */

#include <stdint.h>
#include <sys/types.h>

#define HDC_MAX_DIMENSION                            10048
#define HDC_MIN_DIMENSION                            64
#define HDC_MAX_BINARY_WORDS                             (HDC_MAX_DIMENSION / 64)
#define HDC_MAX_CLASSES                                             128
#define HDC_BITS_TO_WORDS(bits) ((bits + 63) / 64)

struct hdc_binary_classifier {
    int dimension;
    int32_t accum[HDC_MAX_CLASSES][HDC_MAX_DIMENSION];
    uint64_t vector[HDC_MAX_CLASSES][HDC_MAX_BINARY_WORDS];
    int class_count;
};
void hdc_init_binary(unsigned int seed);

void shuffle_binary(int *array, int shuffle_amount);
int classify_binary(struct hdc_binary_classifier *clf, uint64_t *vector);
void train_binary(struct hdc_binary_classifier *clf, uint64_t *vector, int classnum);
void hdc_classifier_init_binary(struct hdc_binary_classifier *clf, int dimension);
void build_prototypes_binary(struct hdc_binary_classifier *clf);
void random_binary(uint64_t *vector, int dimension);
void bind_binary(uint64_t *result_vector, uint64_t *vectora, uint64_t *vectorb, int dimension);
void bundle_binary(uint64_t *result_vector, uint64_t **vectors, int count, int dimension);
void permute_binary(uint64_t *vector, int shift_amount, uint64_t *result_vector,  int dimension);
void similize_binary(int *similarity_int, uint64_t *vectora, uint64_t *vectorb, int dimension);
void zero_vector_binary(uint64_t *vector, int dimension);
void copy_vector_binary(uint64_t *dest, uint64_t *src, int dimension);
void level_encode_binary(float value, uint64_t *result, int dimension);
void id_level_encode_binary(float *values, uint64_t **ids, int channel_amount, uint64_t *result_vector, int dimension);



#endif
