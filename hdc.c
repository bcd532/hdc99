#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void hdc_init(unsigned int seed){srand(seed);}

/* Returns 1 if dimension exceeds max, 0 otherwise */
int check_max(int dimension)
{
    if (dimension <= 0){
        printf("WARNING: dimension (%d) must be greater than 0\n", dimension);
    return 1;}
    if (dimension > HDC_MAX_DIMENSION){
        printf("WARNING: dimension (%d) is above max (%d)\n", dimension, HDC_MAX_DIMENSION);
    return 1;}
    return 0;
    }

/* Fill a vector with random bipolar values (-1.0 or 1.0) */
void random_bipolar(float *vector, int dimension)
{
    if (check_max(dimension)) return;

    for (int i = 0; i < dimension; i++)
    {
        vector[i] = (rand() % 2 == 0) ? 1.0 : -1.0f;
    }
}

/* Bind two vectors via element-wise multiplication */
void bind(float *result, float *vectora, float *vectorb, int dimension)
{
    if (check_max(dimension)) return;

    for (int i = 0; i < dimension; i++)
    {
        result[i] = vectora[i] * vectorb[i];
    }
}
/*
 * WILL COME BACK TO IN ABOUT IDK AN HOUR?
 *
void feature_pair_encode(float *values, float **id_vectors, int feature_count, float *result, int dimension){
    if(check_max(dimension)) return;
    float temp_vector[dimension];
    float temp_vector2[dimension];
    zero_vector(result, dimension);

    for (int i = 0; i < feature_count; i++){
        level_encode(values[i], temp_vector, dimension);
        level_encode(values[i], temp_vector2, dimension);
        bind(temp_vector2, temp_vector, id_vectors[i], dimension);
        for (int j = 0; j < dimension; j++){
                    bundle(result, &id_vectors[i], feature_count, dimension);
        }

    }
}
*/
/* Copy a vector into another */
void copy_vector(float *dest, float *src, int dimension)
{
    if (check_max(dimension)) return;
    for (int i = 0; i < dimension; i++)
    {
        dest[i] = src[i];
    }
}


/* Fill a vector with all zeros */
void zero_vector(float *vector, int dimension)
{
    if (check_max(dimension)) return;

    for (int i = 0; i < dimension; i++)
    {
        vector[i] = 0.0f;
    }
}

/* Bundle multiple vectors via element-wise addition */
void bundle(float *result, float **vectors, int count, int dimension)
{
    if (check_max(dimension)) return;
    zero_vector(result, dimension);
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            result[j] += vectors[i][j];
        }
    }
}

/* Normalize a vector to unit length (L2 norm) */
void normalize(float *target_vector, int dimension)
{
    if (check_max(dimension)) return;

    float sum = 0;

    for (int i = 0; i < dimension; i++){
        sum += target_vector[i] * target_vector[i];}

    float sqrt_array_vector = sqrtf(sum);
    if (sqrt_array_vector == 0) return;

    for (int i = 0; i < dimension; i++){
        target_vector[i] = target_vector[i] / sqrt_array_vector;}
}

/* Compute cosine similarity between two vectors (normalizes both first) */
void similize(float *similar_vector, float *vectora, float *vectorb, int dimension)
{
    if (check_max(dimension)) return;

    float vectora_copy[dimension];
    float vectorb_copy[dimension];
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
    if (check_max(dimension)) return;
    for (int i = 0; i < dimension; i++)
    {
        result[((i + shift_amount) % dimension + dimension) % dimension] = vector[i];
    }
}

/* Train a classifier by adding a vector to a class prototype */
void train(struct hdc_classifier *clf, float *vector, int classnum)
{
    if (classnum >= MAX_CLASSES || classnum < 0 ) return;

    for (int i = 0; i < clf->dimension; i++)
    {
        clf->vector[classnum][i] += vector[i];
    }
    if (classnum >= clf->class_count) clf->class_count = classnum + 1;
}

void hdc_classifier_init(struct hdc_classifier *clf, int dimension){
    clf->class_count = 0;
    clf->dimension = dimension;
    for (int i = 0; i < MAX_CLASSES; i++)
    {zero_vector(clf->vector[i], clf->dimension);
    }

}


/* Classify a vector by finding the most similar class prototype */
int classify(struct hdc_classifier *clf, float *new_vector)
{
    float startfloat = -2.0f;
    int bestclass = -1;
    float current_similarity_score;
    float proto_copy[clf->dimension];
    float new_vector_copy[clf->dimension];

    for (int i = 0; i < clf->class_count; i++)
    {
        copy_vector(proto_copy, clf->vector[i], clf->dimension);
        copy_vector(new_vector_copy, new_vector, clf->dimension);
        similize(&current_similarity_score, proto_copy, new_vector_copy, clf->dimension);

        if (current_similarity_score > startfloat)
        {
            startfloat = current_similarity_score;
            bestclass = i;
        }
    }
    return bestclass;
}

/* Encode a sequence of symbols into an ngram fingerprint */
void ngram(float **vectors, int symbol_count, int window_size, float *result_vector, int dimension)
{
    if (check_max(dimension)) return;

    zero_vector(result_vector, dimension);
    float window_accumulator[dimension];
    float perm_vector[dimension];

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

/* Fill a vector with all -1.0 values */
void neg_vector(float *vector, int dimension)
{
    if(check_max(dimension)) return;
    for (int i = 0; i < dimension; i++)
    {
        vector[i] = -1.0f;
    }
}

/* Encode a continuous value (0.0 to 1.0) into an HDC vector */
void level_encode(float value, float *result, int dimension)
{
    if(check_max(dimension)) return;

    neg_vector(result, dimension);
    if ( value < 0){ value = 0;}
    else if (value > 1){value = 1;}
    int flip_amt = value * dimension;

    for (int i = 0; i < flip_amt; i++)
    {
        result[i] = 1.0f;
    }
}

/* Encode multiple sensor channels into one HDC vector */
void id_level_encode(float *values, float **id_vectors, int channel_amount, float *result_vector, int dimension)
{
    if(check_max(dimension)) return;
    float temp_vector[dimension];
    float temp_vector2[dimension];
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
