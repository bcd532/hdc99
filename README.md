# hdc99

standalone hyperdimensional computing library in c. no dependencies beyond the standard library.

## what is hdc?

hyperdimensional computing represents data as high-dimensional vectors (thousands of elements) and manipulates them with simple math. random vectors in high dimensions are nearly orthogonal, so each one is naturally distinct. you combine them with a few basic operations to encode structure, relationships, and order.

## operations

| function | what it does |
|---|---|
| `random_bipolar` | generate a random vector of -1.0 and 1.0 values |
| `bind` | element-wise multiply — encodes relationships, reversible |
| `bundle` | element-wise add — combines multiple vectors, like a vote |
| `normalize` | scale to unit length so similarity comparisons are fair |
| `similize` | cosine similarity — returns how similar two vectors are (-1 to 1) |
| `permute` | shift elements with wraparound — encodes position/order |
| `zero_vector` | fill a vector with zeros |
| `copy_vector` | copy one vector into another |

## usage

drop `hdc.h` and `hdc.c` into your project. include the header and you're good.

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
    // sim ≈ 0.0 (random vectors are nearly orthogonal)
}
```

## compile

```
gcc -o example example.c hdc.c -lm
```

## license

mit
