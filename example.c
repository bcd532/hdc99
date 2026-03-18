#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DIM 4096

int decimal_counter(float a)
{
    float simcopy = a;
    float fractal = fmodf(simcopy, 1.0f);
    int counter = 0;

    while (fractal != 0 && counter < 20){
    simcopy = simcopy * 10;
    fractal = fmodf(simcopy, 1.0f);
    counter += 1;
    }
    return counter;
}

int debug_copied_vectors(float *vec_a, float*vec_b, int dimension)
{
    for (int i = 0; i < dimension; i++)
    {
        if (vec_a[i] != vec_b[i]){return 0;}
    }
    return 1;
}

int main(void)
{
    srand(time(NULL));

    float vec1[DIM];
    float vec2[DIM];
    float vec3[DIM];
    float vec4[DIM];

    float vec7[DIM];
    float vec8[DIM];
    
    float *list[] = {vec1, vec2, vec3, vec4};

    float bound[DIM];
    float pack[DIM];


    random_bipolar(vec1, DIM);
    random_bipolar(vec2, DIM);
    random_bipolar(vec3, DIM);
    random_bipolar(vec4, DIM);

    random_bipolar(vec7, DIM);
    random_bipolar(vec8, DIM);

    copy_vector(vec8, vec7, DIM);

    bind(bound, vec1, vec2, DIM);
    bundle(pack, list, 4, DIM);
    normalize(pack, DIM);
    


    /* Print first 20 values so you can see what happened */
    printf("\n\nVECTOR 1: ");
    for (int i = 0; i < 20; i++){printf("\033[33m" "%.1f" "\033[0m" ",", vec1[i]);}

    printf("\n\nVECTOR 2: ");
    for (int i = 0; i < 20; i++){printf("\033[33m" "%.1f" "\033[0m" ",", vec2[i]);}
    
    printf("\n\nBOUND: ");
    for (int i = 0; i < 20; i++){printf("\033[33m" "%.1f" "\033[0m" ",", bound[i]);}

    printf("\n\nBUNDLE: ");
    for (int i = 0; i < 20; i++){printf("\033[33m" "%.5f" "\033[0m" ",", pack[i]);}
    


    float similarity;
    float simcopy = 0.0f;
    similize(&similarity, vec1, vec2, DIM);
    simcopy += similarity;
    
    printf("\nSIMILARITY: "); 
    printf("\033[33m" "%.*f" "\033[0m",decimal_counter(simcopy) , similarity);
    printf("\n\nVECTOR COPY SUCCES?: ");
    if (debug_copied_vectors(vec7, vec8, DIM)){printf("Succes!\n");}
    else{printf("ERROR!\n");}

}
