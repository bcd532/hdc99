#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ── internal state ────────────────────────────────────────────── */

/* randomized flip order for level_encode - eliminates prefix bias */
static int flip_order[HDC_MAX_DIMENSION];

/* flip order_init checker - */
static int flip_order_initialized = 0;

/* ── helpers ───────────────────────────────────────────────────── */

/* NULL checker - returns 1 if any pointer in the array is NULL */
int check_null(const void **ptrs, int count, const char *func_name){
    for (int i = 0; i < count; i++){
        if(ptrs[i] == NULL){
            printf("WARNING: NULL pointer at arg %d in %s\n", i, func_name);
            return 1;
        }
    }
    return 0;
}

/* Shuffle for int arrays */
void shuffle(int *array,  int shuffle_amt){
    for (int i = shuffle_amt -1; i > 0; i--){
        int j = rand() % (i+1);
        int temp_array = array[i];
        array[i] = array[j];
        array[j] = temp_array;


    }
}

/* Fill a vector with all zeros */
void zero_vector(float *vector, int dimension)

{
    const void *args[] = {vector};
    if (check_null(args, 1, "zero_vector")) return;
    if (check_dimension(dimension)) return;

    for (int i = 0; i < dimension; i++)
    {
        vector[i] = 0.0f;
    }
}

/* Fill a vector with all -1.0 values */
void neg_vector(float *vector, int dimension)
{
    const void *args[] = {vector};
    if (check_null(args, 1, "neg_vector")) return;
    if(check_dimension(dimension)) return;
    for (int i = 0; i < dimension; i++)
    {
        vector[i] = -1.0f;
    }
}

/* Copy a vector into another */
void copy_vector(float *dest, float *src, int dimension)
{
    const void *args[] = {dest, src};
    if (check_null(args, 2, "copy_vector")) return;
    if (check_dimension(dimension)) return;
    for (int i = 0; i < dimension; i++)
    {
        dest[i] = src[i];
    }
}

/* ── setup ─────────────────────────────────────────────────────── */

/* Seed PRNG and generate randomized flip order for level_encode */
void hdc_init(unsigned int seed){
    srand(seed);
    for (int i = 0; i < HDC_MAX_DIMENSION; i++){
        flip_order[i] = i;
    }
    shuffle(flip_order, HDC_MAX_DIMENSION);
    flip_order_initialized =1;
}

/* Returns 1 if dimension exceeds max or is <= 0, 0 otherwise */
int check_dimension(int dimension)
{
    if (dimension <= 0){
        printf("WARNING: dimension (%d) must be greater than 0\n", dimension);
    return 1;}
    if (dimension > HDC_MAX_DIMENSION){
        printf("WARNING: dimension (%d) is above max (%d)\n", dimension, HDC_MAX_DIMENSION);
    return 1;}
    return 0;
    }

/* Initialize a classifier - zeros all prototypes and sets dimension */
void hdc_classifier_init(struct hdc_classifier *clf, int dimension){
    const void *args[] = {clf};
    if (check_null(args, 1, "hdc_classifier_init")) return;
    if(check_dimension(dimension)) return;
    clf->class_count = 0;
    clf->dimension = dimension;
    for (int i = 0; i < MAX_CLASSES; i++)
    {zero_vector(clf->vector[i], clf->dimension);
    }

}

/* ── core operations ───────────────────────────────────────────── */

/* Fill a vector with random bipolar values (-1.0 or 1.0) */
void random_bipolar(float *vector, int dimension)
{
    const void *args[] = {vector};
    if (check_null(args, 1, "random_bipolar")) return;
    if (check_dimension(dimension)) return;

    for (int i = 0; i < dimension; i++)
    {
        vector[i] = (rand() % 2 == 0) ? 1.0 : -1.0f;
    }
}

/* Bind two vectors via element-wise multiplication */
void bind(float *result, float *vectora, float *vectorb, int dimension)
{
    const void *args[] = {result, vectora, vectorb};
    if (check_null(args, 3, "bind")) return;
    if (check_dimension(dimension)) return;

    for (int i = 0; i < dimension; i++)
    {
        result[i] = vectora[i] * vectorb[i];
    }
}

/* Bundle multiple vectors via element-wise addition */
void bundle(float *result, float **vectors, int count, int dimension)
{
    const void *args[] = {result, vectors};
    if (check_null(args, 2, "bundle")) return;
    if (check_dimension(dimension)) return;
    if (count <= 0) return;
    zero_vector(result, dimension);
    for (int i = 0; i < count; i++)
    {
        if (vectors[i] == NULL) continue;
        for (int j = 0; j < dimension; j++)
        {
            result[j] += vectors[i][j];
        }
    }
}

/* Normalize a vector to unit length (L2 norm) - does nothing if zero vector */
void normalize(float *target_vector, int dimension)
{
    const void *args[] = {target_vector};
    if (check_null(args, 1, "normalize")) return;
    if (check_dimension(dimension)) return;

    float sum = 0;

    for (int i = 0; i < dimension; i++){
        sum += target_vector[i] * target_vector[i];}

    float sqrt_array_vector = sqrtf(sum);
    if (sqrt_array_vector == 0) return;

    for (int i = 0; i < dimension; i++){
        target_vector[i] = target_vector[i] / sqrt_array_vector;}
}

/* Compute cosine similarity - makes internal copies, does NOT modify originals */
void similize(float *similar_vector, float *vectora, float *vectorb, int dimension)
{
    const void *args[] = {similar_vector, vectora, vectorb};
    if (check_null(args, 3, "similize")) return;
    if (check_dimension(dimension)) return;

    static float vectora_copy[HDC_MAX_DIMENSION];
    static float vectorb_copy[HDC_MAX_DIMENSION];
    copy_vector(vectora_copy, vectora, dimension);
    copy_vector(vectorb_copy, vectorb, dimension);

    float sum = 0;

    normalize(vectora_copy, dimension);
    normalize(vectorb_copy, dimension);

    for (int i = 0; i < dimension; i++)
    {
        sum += vectora_copy[i] * vectorb_copy[i];
    }
    *similar_vector = sum;
}

/* Permute a vector by shifting elements and wrapping around */
void permute(float *vector, int shift_amount, float *result, int dimension)
{
    const void *args[] = {vector, result};
    if (check_null(args, 2, "permute")) return;
    if (check_dimension(dimension)) return;
    for (int i = 0; i < dimension; i++)
    {
        result[((i + shift_amount) % dimension + dimension) % dimension] = vector[i];
    }
}

/* ── encoding ──────────────────────────────────────────────────── */

/* Encode a continuous value (0.0 to 1.0) into an HDC vector.
 * Uses randomized flip order to eliminate prefix bias. */
void level_encode(float value, float *result, int dimension)
{

    const void *args[] = {result};
    if (check_null(args, 1, "level_encode")) return;
    if(check_dimension(dimension)) return;
    if(!flip_order_initialized){
        printf("WARNING: flip_order not initialized!"); return;
    };


    neg_vector(result, dimension);
    if ( value < 0){ value = 0;}
    else if (value > 1){value = 1;}
    int flip_amt = value * dimension;

    for (int i = 0; i < flip_amt; i++)
    {
        result[flip_order[i] % dimension] = 1.0f;
    }
}

/* Encode multiple sensor channels into one HDC vector.
 * Level-encodes each value and binds it with its channel ID. */
void id_level_encode(float *values, float **id_vectors, int channel_amount, float *result_vector, int dimension)
{
    const void *args[] = {values, id_vectors, result_vector};
    if (check_null(args, 3, "id_level_encode")) return;
    if(check_dimension(dimension)) return;
    if (channel_amount <= 0) return;
    static float temp_vector[HDC_MAX_DIMENSION];
    static float temp_vector2[HDC_MAX_DIMENSION];
    zero_vector(result_vector, dimension);
    for (int i = 0; i < channel_amount; i++)
    {
    level_encode(values[i], temp_vector, dimension);
    bind(temp_vector2, temp_vector, id_vectors[i], dimension);
    for (int j = 0; j < dimension; j++)
    {
        result_vector[j] += temp_vector2[j];
    }
    }
}

/* Encode a sequence of symbols into an ngram fingerprint */
void ngram(float **vectors, int symbol_count, int window_size, float *result_vector, int dimension)
{
    const void *args[] = {vectors, result_vector};
    if (check_null(args, 2, "ngram")) return;
    if (check_dimension(dimension)) return;
    if (window_size > symbol_count || window_size <= 0) return;


    zero_vector(result_vector, dimension);
    static float window_accumulator[HDC_MAX_DIMENSION];
    static float perm_vector[HDC_MAX_DIMENSION];

    for (int i = 0; i <= (symbol_count - window_size); i++)
    {
        for (int j = 0; j < window_size; j++)
        {
            permute(vectors[i+j], j, perm_vector, dimension);
            if (j == 0){copy_vector(window_accumulator, perm_vector, dimension);}
            else{bind(window_accumulator, window_accumulator, perm_vector, dimension);}
        }
        for ( int k = 0; k < dimension; k++)
        {
            result_vector[k] += window_accumulator[k];
        }
    }
}

/*
 * Archived until further notice.
void feature_pair_encode(float *values, float **id_vectors, int feature_count, float *result, int dimension){
    if(check_dimension(dimension)) return;
    float temp_vector[dimension];
    float temp_vector2[dimension];
    zero_vector(result, dimension);

    for (int i = 0; i < feature_count; i++){
        level_encode(values[i], temp_vector, dimension);
        bind(temp_vector2, temp_vector, id_vectors[i], dimension);

        for (int d = 0; d < dimension; d++){
            result[d] += temp_vector2[d];
        }

    for (int j  = i + 1; j < feature_count; j++){
        level_encode(values[i], temp_vector, dimension);
        level_encode(values[j], temp_vector2, dimension);
        bind(temp_vector, temp_vector, temp_vector2, dimension);
        for (int d = 0; d < dimension; d++){
            result[d] += temp_vector[d];
            }
        }
    }
}
*/

/* ── classification ────────────────────────────────────────────── */

/* Train a classifier by adding a vector to a class prototype.
 * Silently returns if classnum is out of bounds. */
void train(struct hdc_classifier *clf, float *vector, int classnum)
{
    const void *args[] = {clf, vector};
    if (check_null(args, 2, "train")) return;
    if (classnum >= MAX_CLASSES || classnum < 0 ) return;

    for (int i = 0; i < clf->dimension; i++)
    {
        clf->vector[classnum][i] += vector[i];
    }
    if (classnum >= clf->class_count) clf->class_count = classnum + 1;
}

/* Classify a vector by finding the most similar class prototype.
 * Returns -1 if no classes have been trained. */
int classify(struct hdc_classifier *clf, float *new_vector)
{
    const void *args[] = {clf, new_vector};
    if (check_null(args, 2, "classify")) return -1;

    float startfloat = -2.0f;
    int bestclass = -1;
    float current_similarity_score;

    for (int i = 0; i < clf->class_count; i++)
    {
        similize(&current_similarity_score, clf->vector[i], new_vector, clf->dimension);

        if (current_similarity_score > startfloat)
        {
            startfloat = current_similarity_score;
            bestclass = i;
        }
    }
    return bestclass;
}

void vector_to_complex(float *vector, struct complex_number *output, int dimension){
    const void *args[] = {vector, output};
    if (check_null(args, 2, "vector_to_complex")) return;
    if (check_dimension(dimension)) return;
    for (int i = 0; i < dimension; i++){
        output[i].real = vector[i];
        output[i].imag = 0.0f;
    }
}

int bit_reverse(int index, int num_bits){
    int result = 0;
    for (int i = 0; i < num_bits; i++){
        result <<= 1;
        result |= (index & 1);
        index >>= 1;
    }
    return result;
}

void circular_convolve(float *result, float *vectora, float *vectorb, int dimension){
    const void *args[] = {result, vectora, vectorb};
    if (check_null(args, 3, "circular_convolve")) return;
    if (check_dimension(dimension)) return;
    if ((dimension & (dimension - 1)) != 0){
        printf("WARNING: circular_convolve requires power-of-2 dimension, got %d\n", dimension); return;
    }
    static struct complex_number inputa[HDC_MAX_DIMENSION];
    static struct complex_number inputb[HDC_MAX_DIMENSION];
    static struct complex_number result_cmplx[HDC_MAX_DIMENSION];

    vector_to_complex(vectora, inputa, dimension);
    vector_to_complex(vectorb, inputb, dimension);

    rearrange_complex(inputa, dimension);
    rearrange_complex(inputb, dimension);

    fft(inputa, dimension);
    fft(inputb, dimension);

    for (int i = 0; i < dimension; i++){
        result_cmplx[i] = complex_multiply(inputa[i], inputb[i]);
    }
    rearrange_complex(result_cmplx, dimension);
    inverse_fft(result_cmplx, dimension);
    complex_to_vector(result, result_cmplx, dimension);

}

void inverse_fft(struct complex_number *cmplx, int dimension){
    const void *args[] = {cmplx};
    if (check_null(args, 1, "inverse_fft")) return;
    if ((dimension & (dimension - 1)) != 0){
        printf("WARNING: inverse_fft requires power-of-2 dimension, got %d\n", dimension); return;
    }
    for (int size = 2; size <= dimension; size *= 2){
        for (int i = 0; i < dimension; i += size){
            for (int j = 0; j < size /2; j++){
                float angle = +2.0f * 3.14159265f * j /size;
                struct complex_number twiddle;
                twiddle.real = cosf(angle);
                twiddle.imag = sinf(angle);

                struct complex_number top = cmplx[i + j];
                struct complex_number t = complex_multiply(twiddle, cmplx[i + j  + size /2]);

                cmplx[i + j].real = top.real + t.real;
                cmplx[i +j].imag = top.imag + t.imag;

                cmplx[i + j + size/2].real = top.real - t.real;
                cmplx[i + j + size/2].imag= top.imag - t.imag;

            }
        }
    }
    for (int i = 0; i < dimension; i++){
        cmplx[i].real /= dimension;
        cmplx[i].imag /= dimension;
    }
}

void fft(struct complex_number *cmplx, int dimension){
    const void *args[] = {cmplx};
    if (check_null(args, 1, "fft")) return;
    if ((dimension & (dimension - 1)) != 0){
        printf("WARNING: fft requires power-of-2 dimension, got %d\n", dimension); return;
    }
    for (int size = 2; size <= dimension; size *= 2){
        for (int i = 0; i < dimension; i += size){
            for (int j = 0; j < size /2; j++){
                float angle = -2.0f * 3.14159265f * j /size;
                struct complex_number twiddle;
                twiddle.real = cosf(angle);
                twiddle.imag = sinf(angle);

                struct complex_number top = cmplx[i + j];
                struct complex_number t = complex_multiply(twiddle, cmplx[i + j  + size /2]);

                cmplx[i + j].real = top.real + t.real;
                cmplx[i +j].imag = top.imag + t.imag;

                cmplx[i + j + size/2].real = top.real - t.real;
                cmplx[i + j + size/2].imag= top.imag - t.imag;

            }
        }
    }
}

void rearrange_complex(struct complex_number *cmplx, int dimension ){
    const void *args[] = {cmplx};
    if (check_null(args, 1, "rearrange_complex")) return;
    if (check_dimension(dimension)) return;
    int num_bits = 0;
    int temp = dimension;
    while (temp > 1){
        temp /= 2;
        num_bits++;
    }
    for (int i = 0; i < dimension; i++){
        int rev = bit_reverse(i, num_bits);
        if (rev > i){
            struct complex_number temp_s = cmplx[i];
            cmplx[i] = cmplx[rev];
            cmplx[rev] = temp_s;
        }
    }
}

void complex_to_vector(float *output_vector, struct complex_number *input, int dimension){
    const void *args[] = {output_vector, input};
    if (check_null(args, 2, "complex_to_vector")) return;
    if (check_dimension(dimension)) return;
    for (int i = 0; i < dimension; i++){
        output_vector[i] = input[i].real;
    }
}

struct complex_number complex_multiply(struct complex_number a, struct complex_number b){
    struct complex_number result;
    result.real = (a.real * b.real) - (a.imag * b.imag);
    result.imag = (a.real * b.imag) + (a.imag * b.real);
    return result;
}
