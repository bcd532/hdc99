#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>

#define DIM 512
#define MAX_SAMPLES 200
#define FEATURES 13

struct sample{
    float features[FEATURES];
    int label;
};

void shuffle_sample(struct sample *sample, int count){
    for (int i = count -1; i>0; i--){
        int j = rand() % (i+ 1);
        struct sample temp = sample[i];
        sample[i] = sample[j];
        sample[j] = temp;
    }
}

void dim_limiter(int dimension, int new_dimension);


int main(){
    FILE *file = fopen("wine.data", "r");
    hdc_init(20);
    char line[512];
    int count = 0;
    static float id_storage[FEATURES][DIM];
    float *ids[FEATURES];
    struct sample data[MAX_SAMPLES];

    for (int n = 0 ; n < FEATURES; n++){
        random_bipolar(id_storage[n], DIM);
        ids[n] = id_storage[n];
    }




    while(fgets(line, sizeof(line),file) && count < MAX_SAMPLES){
        int label;
        float feat[13];
        sscanf(
        line,
        "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
        &label, &feat[0], &feat[1],&feat[2], &feat[3], &feat[4],
        &feat[5], &feat[6], &feat[7], &feat[8], &feat[9], &feat[10],
        &feat[11], &feat[12]);
        data[count].label=label;
        for (int i = 0; i < FEATURES; i++){
            data[count].features[i] = feat[i];
        }
count++;
    }
    shuffle_sample(data, count);
    int train_count = count *.8;
    float feat_min[FEATURES];
    float feat_max[FEATURES];

    for (int k = 0; k < FEATURES; k++){
        feat_min[k] = data[0].features[k];
        feat_max[k] = data[0].features[k];
    }

    for (int i= 0; i < train_count; i++){
        for (int f = 0; f < FEATURES; f++){
            if (data[i].features[f] < feat_min[f]) feat_min[f] = data[i].features[f];
            if (data[i].features[f] > feat_max[f]) feat_max[f] = data[i].features[f];
        }
    }

        static struct hdc_classifier clf;
        hdc_classifier_init(&clf, DIM);

        for (int z = 0; z < train_count; z++){
            float scaled[FEATURES];
            for (int f = 0; f < FEATURES; f++){


            if (feat_max[f] == feat_min[f]) scaled[f] = .5f;
            else scaled[f] = (data[z].features[f] - feat_min[f]) / (feat_max[f] - feat_min[f]);
            }
            float encoded[DIM];
            id_level_encode(scaled, ids, FEATURES, encoded, DIM);
            train(&clf, encoded, data[z].label);

    }
        int correct = 0;
    for (int e = train_count; e < count; e++){
        float scaled[FEATURES];

        for (int p = 0; p < FEATURES; p++){

            if (feat_max[p] == feat_min[p]) scaled[p] = .5f;
            else scaled[p] = (data[e].features[p] - feat_min[p]) / (feat_max[p] - feat_min[p]);
        }

            float encoded[DIM];
            id_level_encode(scaled, ids, FEATURES, encoded, DIM);
            int predicted = classify(&clf, encoded);
            if (predicted == data[e].label) correct++;
        }
    int test_count = count - train_count;
    printf("accuracy : %.1f%% (%d/%d)\n",
        (float)correct / test_count* 100.0f, correct,test_count);
    }
