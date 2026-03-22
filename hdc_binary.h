#ifndef HDC_BINARY_H
#define HDC_BINARY_H

/**
 * hdc_binary.h - Hyperdimensional computing library (binary version).
 * High-dimensional vector operations for encoding and comparing data using binary vectors.
 */

#include <stdint.h>

#define HDC_MAX_DIMENSION                            10048
#define HDC_MIN_DIMENSION                            64
#define HDC_MAX_BINARY_WORDS                             (HDC_MAX_DIMENSION / 64)
#define HDC_MAX_CLASSES                                             128
#define HDC_BITS_TO_WORDS(bits) ((bits + 63) / 64)


void random_binary(uint64_t *vector, int dimension);
void bind_binary(uint64_t *result_vector, uint64_t *vectora, uint64_t *vectorb, int dimension);
void bundle_binary(uint64_t *result_vector, uint64_t **vectors, int count, int dimension);
void permute_binary(uint64_t *vector, int shift_amount, uint64_t *result_vector,  int dimension);
void similize_binary(int *similarity_int, uint64_t *vectora, uint64_t *vectorb, int dimension);

#endif
