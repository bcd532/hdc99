#ifndef HDC_H
#define HDC_H

/**
 * hdc.h - hyperdimensional computing library
 * high-dimensional vector operations for encoding and comparing data.
 * two files: hdc.h + hdc.c. zero dependencies beyond the standard library.
 *
 * compile: gcc -std=c99 -I. -o app your_file.c hdc.c -lm
 */

#define HDC_VERSION        "0.2.1"
#define HDC_MAX_DIMENSION  10048
#define MAX_CLASSES        128

/* ── classifier ────────────────────────────────────────────────────
 *
 * WARNING: this struct is ~5MB. do NOT declare as a local variable.
 * use static or global:
 *
 *   static struct hdc_classifier clf;
 *   hdc_classifier_init(&clf, 4096);
 */
struct hdc_classifier {
    int dimension;
    float vector[MAX_CLASSES][HDC_MAX_DIMENSION];
    int class_count;
};

struct complex_number{
   float real;
  float imag;
};

/* ── setup ─────────────────────────────────────────────────────── */

/**
 * Seed the random number generator.
 * Call once before using random_bipolar.
 * @param seed  seed value (use a fixed number for reproducibility)
 */
void hdc_init(unsigned int seed);

/**
 * Initialize a classifier. Zeros all prototypes and sets dimension.
 * @param clf        pointer to the classifier
 * @param dimension  number of elements per vector
 */
void hdc_classifier_init(struct hdc_classifier *clf, int dimension);

/**
 * Returns 1 if dimension is invalid (<=0 or > HDC_MAX_DIMENSION).
 * Prints a warning. Used internally by all functions.
 * @param dimension  dimension to validate
 * @return           1 if bad, 0 if ok
 */
int check_dimension(int dimension);

/**
 * Returns 1 if any pointer in the array is NULL.
 * Prints a warning with the function name and argument index.
 * @param ptrs       array of pointers to check
 * @param count      number of pointers in the array
 * @param func_name  name of the calling function (for the warning)
 * @return           1 if any is NULL, 0 if all ok
 */
int check_null(const void **ptrs, int count, const char *func_name);

/* ── core operations ───────────────────────────────────────────── */

/* Broken as of 9:16am 03-22-2026 (made accuracy worse on iris, ionosphere, and wine benchmarks)
 * will be saving for later to see if improvements can be made!

void feature_pair_encode(float *values, float **id_vectors, int feature_count, float *result, int dimension);
*/


/**
 * Fill a vector with random bipolar values (-1.0 or 1.0).
 * Call hdc_init() first or results will be the same every run.
 * @param vector     output vector to fill
 * @param dimension  number of elements in the vector
 */
void random_bipolar(float *vector, int dimension);

/**
 * Bind two vectors via element-wise multiplication.
 * Encodes relationships between two concepts. Reversible.
 * @param result     output vector
 * @param vectora    first input vector
 * @param vectorb    second input vector
 * @param dimension  number of elements in each vector
 */
void bind(float *result, float *vectora, float *vectorb, int dimension);

/**
 * Bundle multiple vectors via element-wise addition.
 * Combines multiple representations into one, like a vote.
 * @param result     output vector (zeroed internally before use)
 * @param vectors    array of pointers to input vectors
 * @param count      number of vectors to bundle
 * @param dimension  number of elements in each vector
 */
void bundle(float *result, float **vectors, int count, int dimension);

/**
 * Normalize a vector to unit length (L2 norm).
 * Scales the vector so similarity comparisons are fair.
 * Does nothing if the vector is all zeros.
 * @param target_vector  vector to normalize in place
 * @param dimension      number of elements in the vector
 */
void normalize(float *target_vector, int dimension);

/**
 * Compute cosine similarity between two vectors.
 * Makes internal copies - does NOT modify the originals.
 * Result: 1.0 = identical, 0.0 = unrelated, -1.0 = opposite.
 * @param similar_vector  pointer to store the result
 * @param vector_a        first input vector
 * @param vector_b        second input vector
 * @param dimension       number of elements in each vector
 */
void similize(float *similar_vector, float *vector_a, float *vector_b, int dimension);

/**
 * Permute a vector by shifting elements and wrapping around.
 * Encodes positional/order information.
 * @param vector        input vector
 * @param shift_amount  number of positions to shift right
 * @param result        output vector (must not be the same as input)
 * @param dimension     number of elements in each vector
 */
void permute(float *vector, int shift_amount, float *result, int dimension);

/* ── encoding ──────────────────────────────────────────────────── */

/**
 * Encode a sequence of symbol vectors into an ngram fingerprint.
 * Captures pattern and order information via sliding window.
 * @param vectors        array of pointers to symbol vectors
 * @param symbol_count   number of symbols in the sequence
 * @param window_size    ngram window size (e.g. 3 for trigrams)
 * @param result_vector  output vector
 * @param dimension      number of elements in each vector
 */
void ngram(float **vectors, int symbol_count, int window_size, float *result_vector, int dimension);

/**
 * Encode a continuous value (0.0 to 1.0) into an HDC vector.
 * Nearby values produce similar vectors, distant values produce orthogonal ones.
 * Values outside 0.0-1.0 are clamped automatically.
 * @param value      input value (clamped to 0.0-1.0)
 * @param result     output vector
 * @param dimension  number of elements in the vector
 */
void level_encode(float value, float *result, int dimension);

/**
 * Encode multiple sensor channels into one HDC vector.
 * Level-encodes each value and binds it with its channel ID.
 * @param values          array of sensor readings (0.0 to 1.0)
 * @param id_vectors      array of pointers to channel ID vectors
 * @param channel_amount  number of sensor channels
 * @param result_vector   output vector
 * @param dimension       number of elements in each vector
 */
void id_level_encode(float *values, float **id_vectors, int channel_amount, float *result_vector, int dimension);

/* ── classification ────────────────────────────────────────────── */

/**
 * Train a classifier by adding a vector to a class prototype.
 * Call multiple times per class to build a stronger prototype.
 * Silently returns if classnum is out of bounds.
 * @param clf       pointer to the classifier
 * @param vector    training example vector
 * @param classnum  which class (0 to MAX_CLASSES-1)
 */
void train(struct hdc_classifier *clf, float *vector, int classnum);

/**
 * Classify a vector by finding the most similar class prototype.
 * Returns -1 if no classes have been trained.
 * @param clf         pointer to the classifier
 * @param new_vector  vector to classify
 * @return            class number of the best match, or -1
 */
int classify(struct hdc_classifier *clf, float *new_vector);

/* ── helpers ───────────────────────────────────────────────────── */

/**
 * Fill a vector with all zeros.
 * @param vector     vector to zero out
 * @param dimension  number of elements in the vector
 */
void zero_vector(float *vector, int dimension);

/**
 * Fisher-Yates shuffle for int arrays.
 * @param array       array to shuffle in place
 * @param shuffle_amt number of elements in the array
 */
void shuffle(int *array, int shuffle_amt);

/**
 * Fill a vector with all -1.0 values.
 * @param vector     vector to fill
 * @param dimension  number of elements in the vector
 */
void neg_vector(float *vector, int dimension);

/**
 * Copy a vector into another.
 * @param dest       destination vector
 * @param src        source vector
 * @param dimension  number of elements in each vector
 */
void copy_vector(float *dest, float *src, int dimension);

struct complex_number complex_multiply(struct complex_number a, struct complex_number b);

void rearrange_complex(struct complex_number *cmplx, int dimension);

void fft(struct complex_number *cmplx, int dimension);
void inverse_fft(struct complex_number *cmplx, int dimension);
int bit_reverse(int index, int num_bits);
void circular_convolve(float *result, float *vectora, float *vectorb, int dimension);
void complex_to_vector(float *output_vector, struct complex_number *input, int dimension);


void vector_to_complex(float *vector, struct complex_number *output, int dimension);



#endif
