# hdc99

standalone hyperdimensional computing framework in c. two files, zero dependencies.

drop `hdc.h` and `hdc.c` into your project and you have on-device pattern recognition.

## what is hdc?

hyperdimensional computing represents data as high-dimensional vectors (thousands of elements) and manipulates them with simple math. random vectors in high dimensions are nearly orthogonal, so each one is naturally distinct. you combine them with a few basic operations to encode structure, relationships, and order.

## operations

### primitives

| function | what it does |
|---|---|
| `random_bipolar` | generate a random vector of -1.0 and 1.0 values |
| `bind` | element-wise multiply — encodes relationships, reversible |
| `bundle` | element-wise add — combines multiple vectors, like a vote |
| `normalize` | scale to unit length so similarity comparisons are fair |
| `similize` | cosine similarity — returns how similar two vectors are (-1 to 1) |
| `permute` | shift elements with wraparound — encodes position/order |
| `zero_vector` | fill a vector with zeros |
| `neg_vector` | fill a vector with -1.0 values |
| `copy_vector` | copy one vector into another |

### encoding

| function | what it does |
|---|---|
| `ngram` | encode a sequence into a fingerprint that captures patterns and order |
| `level_encode` | encode a continuous value (0.0-1.0) into an hdc vector |
| `id_level_encode` | encode multiple sensor channels into one vector with channel identity |

### classification

| function | what it does |
|---|---|
| `train` | add a training example to a class prototype |
| `classify` | find the most similar class for a new vector |

## usage

```c
#include "hdc.h"

#define DIM 4096

int main(void)
{
    float a[DIM], b[DIM], result[DIM];

    random_bipolar(a, DIM);
    random_bipolar(b, DIM);

    bind(result, a, b, DIM);

    float sim;
    similize(&sim, a, b, DIM);
    // sim ~ 0.0 (random vectors are nearly orthogonal)
}
```

## compile

```
gcc -I. -o example examples/example.c hdc.c -lm
```

## license

mit
