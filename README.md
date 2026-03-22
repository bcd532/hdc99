# hdc99

standalone hyperdimensional computing library in c99. two files, zero dependencies, runs anywhere.

drop `hdc.h` and `hdc.c` into your project. that's it.

## what is this

hdc encodes data into high-dimensional vectors and classifies by similarity. no training loop, no backpropagation, no gpu. works on microcontrollers, laptops, bare metal, whatever.

tested on real datasets:
- **wine** (13 features, 3 classes): 100% accuracy
- **iris** (4 features, 3 classes): 93-100% accuracy
- **ionosphere** (34 features, 2 classes): 90-93% accuracy

all under 5ms total (train + classify). comparable to SVM accuracy at a fraction of the compute.

## what you get

**primitives** - the core hdc algebra
- `bind` - element-wise multiply, encodes relationships, reversible
- `bundle` - element-wise add, combines vectors like a vote
- `permute` - circular shift, encodes position and order
- `normalize` - scale to unit length for fair comparisons
- `similize` - cosine similarity between two vectors
- `random_bipolar` - generate random vectors of -1 and 1

**encoding** - turn real data into vectors
- `level_encode` - continuous value (0.0-1.0) to vector, randomized flip order for zero bias
- `id_level_encode` - multi-channel sensor data to one vector with channel identity
- `ngram` - sequence fingerprinting for pattern and order capture

**classification** - train and predict
- `train` - add examples to class prototypes
- `classify` - find the most similar class, returns -1 if nothing trained

**helpers** - `zero_vector`, `neg_vector`, `copy_vector`, `shuffle`

## quick start

```c
#include "hdc.h"

#define DIM 4096

int main(void)
{
    hdc_init(42);  // always call this first

    float a[DIM], b[DIM], result[DIM];
    random_bipolar(a, DIM);
    random_bipolar(b, DIM);

    bind(result, a, b, DIM);

    float sim;
    similize(&sim, a, b, DIM);
    // sim is near 0.0 - random vectors are nearly orthogonal
}
```

## compile

```
gcc -std=c99 -I. -o app your_file.c hdc.c -lm
```

## warnings

- **call `hdc_init()` before anything else.** level_encode uses a randomized internal table that gets built during init. skip it and you get worse encoding with no error.
- **the classifier struct is ~5MB.** declare it `static` or global, never as a local variable inside a function. stack overflow otherwise.
  ```c
  static struct hdc_classifier clf;
  hdc_classifier_init(&clf, 4096);
  ```
- **all functions do NULL and bounds checking.** you'll get a printed warning instead of a segfault if you pass bad pointers or invalid dimensions.
- **max dimension is 10048.** configurable via `HDC_MAX_DIMENSION` in the header.

## what's next

- binary hdc (xor bind, hamming distance, bit-packed vectors - way faster, way less memory)
- circular convolution binding for richer cross-feature encoding
- online/adaptive classifier that corrects itself during inference

## license

mit
