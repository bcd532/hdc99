#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Returns 1 if dimension exceeds max, 0 otherwise */
int check_max(int dimension)
{
    if (dimension > HDC_MAX_DIMENSION)
    {printf(
            "\033[31m" 
            "WARNING: Your dimension (%d) is above dimension max (%d), reduce to continue." 
            "\033[0m",

            dimension,
            HDC_MAX_DIMENSION); 
    return 1;
    }
    else{return 0;
    }
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

/* Copy a vector into another */
void copy_vector(float *dest, float *src, int dimension)
{
    if (check_max(dimension)) return;
    zero_vector(dest, dimension);
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

    for (int i = 0; i < dimension; i++)
    {
        sum += target_vector[i] * target_vector[i];
    }
    float sqrt_array_vector = sqrtf(sum);
    for (int i = 0; i < dimension; i++)
    {
        target_vector[i] = target_vector[i] / sqrt_array_vector;
    }
}

/* Compute cosine similarity between two vectors (normalizes both first) */
void similize(float *similar_vector, float *vectora, float *vectorb, int dimension)
{
    if (check_max(dimension)) return;

    float sum = 0;
    
    normalize(vectora, dimension);
    normalize(vectorb, dimension);
    for (int i = 0; i < dimension; i++)
    {
        sum += vectora[i] * vectorb[i];
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
    for (int i = 0; i < clf->dimension; i++)
    {
        clf->vector[classnum][i] += vector[i];
    }
}

/* Classify a vector by finding the most similar class prototype */
int classify(struct hdc_classifier *clf, float *new_vector)
{
    float startfloat = -2.0f;
    int bestclass;
    float current_similarity_score;
    float proto_copy[HDC_MAX_DIMENSION];
    float new_vector_copy[HDC_MAX_DIMENSION];

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
    float window_accumulator[HDC_MAX_DIMENSION];
    float perm_vector[HDC_MAX_DIMENSION];

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
    for (int i = 0; i < dimension; i++)
    {
        vector[i] = -1.0f;
    }
}

/* Encode a continuous value (0.0 to 1.0) into an HDC vector */
void level_encode(float value, float *result, int dimension)
{
    neg_vector(result, dimension);

    int flip_amt = value * dimension;

    for (int i = 0; i < flip_amt; i++)
    {
        result[i] = 1.0f;
    }
}


