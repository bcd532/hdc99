#ifndef HDC_H
#define HDC_H

/**
 * hdc.h - Hyperdimensional computing library.
 * High-dimensional vector operations for encoding and comparing data.
 */

#define HDC_MAX_DIMENSION     10048
#define MAX_CLASSES           128


struct hdc_classifier{
    int dimension;
    float vector[MAX_CLASSES][HDC_MAX_DIMENSION];
    int class_count;
};

int check_max(int dimension);

/**
 * Fill a vector with random bipolar values (-1.0 or 1.0).
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
 * @param result     output vector
 * @param vectors    array of pointers to input vectors
 * @param count      number of vectors to bundle
 * @param dimension  number of elements in each vector
 */
void bundle(float *result, float **vectors, int count, int dimension);

/**
 * Normalize a vector to unit length (L2 norm).
 * Scales the vector so similarity comparisons are fair.
 * @param target_vector  vector to normalize in place
 * @param dimension      number of elements in the vector
 */
void normalize(float *target_vector, int dimension);

/**
 * Compute cosine similarity between two vectors.
 * Normalizes both vectors, then returns their dot product.
 * Result: 1.0 = identical, 0.0 = unrelated, -1.0 = opposite.
 * @param similar_vector  pointer to store the result
 * @param vector_a        first input vector (will be normalized)
 * @param vector_b        second input vector (will be normalized)
 * @param dimension       number of elements in each vector
 */
void similize(float *similar_vector, float *vector_a, float *vector_b, int dimension);

/**
 * Permute a vector by shifting elements and wrapping around.
 * Encodes positional/order information.
 * @param vector        input vector
 * @param shift_amount  number of positions to shift right
 * @param result        output vector
 * @param dimension     number of elements in each vector
 */
void permute(float *vector, int shift_amount, float *result, int dimension);

/**
 * Fill a vector with all zeros.
 * Useful for initializing accumulators before bundling.
 * @param vector     vector to zero out
 * @param dimension  number of elements in the vector
 */
void zero_vector(float *vector, int dimension);

/**
 * Copy a vector into another.
 * @param dest       destination vector
 * @param src        source vector
 * @param dimension  number of elements in each vector
 */
void copy_vector(float *dest, float *src, int dimension);

/**
 * Train a classifier by adding a vector to a class prototype.
 * Call multiple times per class to build a stronger prototype.
 * @param clf       pointer to the classifier
 * @param vector    training example vector
 * @param classnum  which class this example belongs to
 */
void train(struct hdc_classifier *clf, float *vector, int classnum);

/**
 * Classify a vector by finding the most similar class prototype.
 * @param clf         pointer to the classifier
 * @param new_vector  vector to classify
 * @return            class number of the best match
 */
int classify(struct hdc_classifier *clf, float *new_vector);

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
 * Fill a vector with all -1.0 values.
 * @param vector     vector to fill
 * @param dimension  number of elements in the vector
 */
void neg_vector(float *vector, int dimension);

/**
 * Encode a continuous value (0.0 to 1.0) into an HDC vector.
 * Nearby values produce similar vectors, distant values produce orthogonal ones.
 * @param value      input value between 0.0 and 1.0
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


#endif
