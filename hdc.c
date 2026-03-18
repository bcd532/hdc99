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
        result[(i + shift_amount) % dimension] = vector[i];
    }
}
