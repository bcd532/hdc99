#include "hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * language recognition benchmark
 * uses character trigrams (the classic HDC demo)
 * compares element-wise bind vs circular convolution
 *
 * compile: gcc -std=c99 -O2 -I. -o lang_benchmark examples/lang_benchmark.c hdc.c -lm
 */

#define DIM 4096
#define NUM_LANGS 5
#define TRIGRAM_WINDOW 3
#define MAX_TEXT 50000
#define MAX_TEST_FILES 200

/* one random vector per byte value (256 possible) */
static float codebook[256][DIM];

/* encode a text string into an HDC vector using character trigrams.
 * uses the provided bind_func to combine trigram elements. */
void encode_text_bind(const char *text, int len, float *result, int dimension)
{
    zero_vector(result, dimension);
    if (len < TRIGRAM_WINDOW) return;

    float perm0[dimension], perm1[dimension], perm2[dimension];
    float bound[dimension];

    for (int i = 0; i <= len - TRIGRAM_WINDOW; i++) {
        unsigned char c0 = (unsigned char)text[i];
        unsigned char c1 = (unsigned char)text[i+1];
        unsigned char c2 = (unsigned char)text[i+2];

        permute(codebook[c0], 0, perm0, dimension);
        permute(codebook[c1], 1, perm1, dimension);
        permute(codebook[c2], 2, perm2, dimension);

        bind(bound, perm0, perm1, dimension);
        bind(bound, bound, perm2, dimension);

        for (int d = 0; d < dimension; d++)
            result[d] += bound[d];
    }
}

void encode_text_conv(const char *text, int len, float *result, int dimension)
{
    zero_vector(result, dimension);
    if (len < TRIGRAM_WINDOW) return;

    float perm0[dimension], perm1[dimension], perm2[dimension];
    float bound[dimension];

    for (int i = 0; i <= len - TRIGRAM_WINDOW; i++) {
        unsigned char c0 = (unsigned char)text[i];
        unsigned char c1 = (unsigned char)text[i+1];
        unsigned char c2 = (unsigned char)text[i+2];

        permute(codebook[c0], 0, perm0, dimension);
        permute(codebook[c1], 1, perm1, dimension);
        permute(codebook[c2], 2, perm2, dimension);

        circular_convolve(bound, perm0, perm1, dimension);
        circular_convolve(bound, bound, perm2, dimension);

        for (int d = 0; d < dimension; d++)
            result[d] += bound[d];
    }
}

/* read entire file into buffer, return length */
int read_file(const char *path, char *buf, int max_len)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int len = fread(buf, 1, max_len - 1, f);
    buf[len] = '\0';
    fclose(f);
    /* lowercase */
    for (int i = 0; i < len; i++)
        if (buf[i] >= 'A' && buf[i] <= 'Z') buf[i] += 32;
    return len;
}

int main(void)
{
    hdc_init(42);

    printf("hdc99 — language recognition benchmark\n");
    printf("dimension: %d, trigram window: %d\n\n", DIM, TRIGRAM_WINDOW);

    /* build codebook — one random vector per byte */
    for (int i = 0; i < 256; i++)
        random_bipolar(codebook[i], DIM);

    /* languages to test (subset for speed) */
    const char *lang_codes[] = {"en", "de", "fr", "es", "it"};
    const char *train_files[] = {
        "/tmp/lang-data/training_texts/english.txt",
        "/tmp/lang-data/training_texts/german.txt",
        "/tmp/lang-data/training_texts/french.txt",
        "/tmp/lang-data/training_texts/spanish.txt",
        "/tmp/lang-data/training_texts/italian.txt"
    };

    static struct hdc_classifier clf_bind;
    static struct hdc_classifier clf_conv;
    hdc_classifier_init(&clf_bind, DIM);
    hdc_classifier_init(&clf_conv, DIM);

    char text_buf[MAX_TEXT];

    /* train on each language's full training text */
    printf("training...\n");
    for (int lang = 0; lang < NUM_LANGS; lang++) {
        int len = read_file(train_files[lang], text_buf, MAX_TEXT);
        if (len == 0) { printf("  error: can't read %s\n", train_files[lang]); continue; }
        printf("  %s: %d chars\n", lang_codes[lang], len);

        float enc_bind[DIM];
        encode_text_bind(text_buf, len, enc_bind, DIM);
        train(&clf_bind, enc_bind, lang);

        float enc_conv[DIM];
        encode_text_conv(text_buf, len, enc_conv, DIM);
        train(&clf_conv, enc_conv, lang);
    }

    /* test on snippets */
    printf("\ntesting...\n");
    int correct_bind = 0, correct_conv = 0, total = 0;
    char path[512];

    for (int lang = 0; lang < NUM_LANGS; lang++) {
        int lang_correct_bind = 0, lang_correct_conv = 0, lang_total = 0;

        for (int t = 0; t < MAX_TEST_FILES; t++) {
            snprintf(path, sizeof(path), "/tmp/lang-data/testing_texts/%s_%d_p.txt",
                     lang_codes[lang], t);

            int len = read_file(path, text_buf, MAX_TEXT);
            if (len < 10) continue;

            float enc_bind[DIM];
            encode_text_bind(text_buf, len, enc_bind, DIM);
            int pred_bind = classify(&clf_bind, enc_bind);

            float enc_conv[DIM];
            encode_text_conv(text_buf, len, enc_conv, DIM);
            int pred_conv = classify(&clf_conv, enc_conv);

            if (pred_bind == lang) { correct_bind++; lang_correct_bind++; }
            if (pred_conv == lang) { correct_conv++; lang_correct_conv++; }
            total++;
            lang_total++;
        }
        if (lang_total > 0) {
            printf("  %s: bind %.1f%% (%d/%d)  conv %.1f%% (%d/%d)\n",
                   lang_codes[lang],
                   (float)lang_correct_bind / lang_total * 100.0f, lang_correct_bind, lang_total,
                   (float)lang_correct_conv / lang_total * 100.0f, lang_correct_conv, lang_total);
        }
    }

    printf("\n--- results ---\n");
    printf("  bind (standard):       %.1f%% (%d/%d)\n",
           (float)correct_bind / total * 100.0f, correct_bind, total);
    printf("  circular convolution:  %.1f%% (%d/%d)\n",
           (float)correct_conv / total * 100.0f, correct_conv, total);

    return 0;
}
