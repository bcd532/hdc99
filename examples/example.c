#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DIM 4096



int test_copy()
{
    float src_vector[DIM];
    float dest_vector[DIM];
    random_bipolar(src_vector, DIM);
    copy_vector(dest_vector, src_vector, DIM);

    for (int i = 0; i < DIM; i++)
    {
        if (src_vector[i] != dest_vector[i]){return 0;}
    }
    return 1;
}

int test_bipolar()
{
    float vectest[DIM];
    random_bipolar(vectest, DIM);

    for (int i = 0; i < DIM; i++)
    {
        if (vectest[i] != 1.0f && vectest[i] != -1.0f)
        {return 0;}
    }
    return 1;
}

int test_bind()
{
    float target_vector[DIM];
    float vector[DIM];

    random_bipolar(vector, DIM);

    bind(target_vector, vector, vector, DIM);

    for (int i =0; i < DIM; i++)
    {
        if (target_vector[i] != 1.0f)
        {return 0;}
    }
    return 1;
}

int test_zero()
{
    float target_vector[DIM];
    zero_vector(target_vector, DIM);
    
    for (int i = 0; i < DIM; i++)
    {
        if (target_vector[i] != 0.0f)
        {return 0;}
    }
    return 1;
}

int test_bundle()
{
    float vector[DIM];
    random_bipolar(vector, DIM);
    float result_vector[DIM];
    float *list[] = {vector, vector, vector};
    bundle(result_vector, list, 3, DIM);

    for (int i = 0; i < DIM; i++)
    {
        if (result_vector[i] != 3.0f && result_vector[i] != -3.0f)
        {return 0;}
    }
    return 1;
}

int test_normalize()
{
    float target_vector[DIM];
    float sum = 0;
    random_bipolar(target_vector, DIM);
    normalize(target_vector, DIM);

    for (int i = 0; i < DIM; i++)
    {
        sum += target_vector[i] * target_vector[i];
    }
    float length = sqrtf(sum);
    if (length < 0.999f || length > 1.001f){return 0;}
    return 1;
}

int test_permute()
    {
    float result[DIM];
    float result2[DIM];
    float target_vector[DIM];
    random_bipolar(target_vector, DIM);
    float copy_vec[DIM];
    

    copy_vector(copy_vec, target_vector, DIM);
    permute(target_vector, 5, result, DIM);
    permute(result, -5, result2, DIM);

    for (int i = 0; i < DIM; i++)
    {
        if (result2[i] != copy_vec[i])
        {return 0;}
    }
    return 1;
}

int test_similize()
{
    float vec[DIM];
    float copy_vec[DIM];
    float similarvector;

    random_bipolar(vec, DIM);
    copy_vector(copy_vec, vec, DIM);

    similize(&similarvector, vec, copy_vec, DIM);
    if (similarvector < 0.999f){return 0;}
    return 1;
}


/*-----------------------------------------------------------*/

int main(void)
{

    printf("[hdc99 tests]\n\n");

    printf("bind ....... %s\n", test_bind() ? "PASS" : "FAIL");
    printf("bipolar ....... %s\n", test_bipolar() ? "PASS" : "FAIL");
    printf("zero ....... %s\n", test_zero() ? "PASS" : "FAIL");
    printf("bundle ....... %s\n", test_bundle() ? "PASS" : "FAIL");
    printf("copy ....... %s\n", test_copy() ? "PASS" : "FAIL");
    printf("normalize ...... %s\n", test_normalize() ? "PASS" : "FAIL");
    printf("permute ...... %s\n", test_permute() ? "PASS" : "FAIL");
    printf("similize ....... %s\n", test_similize() ? "PASS" : "FAIL");

}
