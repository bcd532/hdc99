# hdc99

![C99](https://img.shields.io/badge/C99-00599C?style=flat&logo=c&logoColor=white)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-0.3.2-orange)
![Platform](https://img.shields.io/badge/platform-any-lightgrey)

standalone hyperdimensional computing library in c99. zero dependencies, runs anywhere.

float vectors and binary (bit-packed) vectors. fft-based circular convolution. drop-in ready.

## what is this?

hdc encodes data into high-dimensional vectors and classifies by similarity. no training loop, no backpropagation, no gpu. works on microcontrollers, laptops, bare metal, whatever.

benchmarked on UCI datasets:

| dataset | features | classes | accuracy | notes |
|---------|----------|---------|----------|-------|
| wine | 13 | 3 | **97.0%** avg (200 seeds) | 100% on ~1 in 3 seeds |
| ionosphere | 34 | 2 | **90.3%** avg (200 seeds) | beats raw kNN by 6% |
| iris | 4 | 3 | **96.7%** avg | matches SVM |

- matches torchhd accuracy, runs **16x faster**
- **25,000x lighter** (15KB vs 500MB+)
- works at 64 dimensions just as well as 4096
- first known HDC implementation tested on quantum hardware (IonQ)

## what you get

### float (hdc.h + hdc.c)

**primitives**
- `bind` — element-wise multiply, encodes relationships, reversible
- `bundle` — element-wise add, combines vectors like a vote
- `permute` — circular shift, encodes position and order
- `normalize` — scale to unit length for fair comparisons
- `similize` — cosine similarity between two vectors
- `random_bipolar` — generate random vectors of -1 and 1
- `circular_convolve` — fft-based circular convolution (hrr-style binding)

**encoding**
- `level_encode` — continuous value (0.0-1.0) to vector, randomized flip order for zero bias
- `id_level_encode` — multi-channel sensor data to one vector with channel identity
- `ngram` — sequence fingerprinting for pattern and order capture

**classification**
- `train` — add examples to class prototypes
- `classify` — find the most similar class, returns -1 if nothing trained
- `hdc_classifier_init` — initialize classifier with dimension

**fft**
- `fft` / `inverse_fft` — fast fourier transform on complex arrays
- `circular_convolve` — convolve two vectors via fft (captures cross-feature relationships)
- `complex_multiply` — complex number multiplication
- `vector_to_complex` / `complex_to_vector` — conversion helpers

**helpers** — `zero_vector`, `neg_vector`, `copy_vector`, `shuffle`, `check_null`, `check_dimension`

### binary (hdc_binary.h + hdc_binary.c)

![Binary HDC](https://img.shields.io/badge/binary-bit--packed-purple)

bit-packed vectors in `uint64_t` arrays. 64 dimensions per word. way faster, way less memory.

- `random_binary` — generate random bit vectors
- `bind_binary` — xor binding (single cpu instruction per 64 dims)
- `bundle_binary` — majority vote across multiple vectors
- `similize_binary` — hamming distance via popcount
- `permute_binary` — bit-level circular shift
- `level_encode_binary` — continuous value to binary vector
- `id_level_encode_binary` — multi-channel sensor encoding
- `train_binary` / `classify_binary` — accumulator-based classifier with majority vote thresholding
- `build_prototypes_binary` — threshold accumulators into binary prototypes

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
    // sim is near 0.0 — random vectors are nearly orthogonal
}
```

## compile

```
gcc -std=c99 -I. -o app your_file.c hdc.c -lm
```

for binary hdc:
```
gcc -std=c99 -I. -o app your_file.c hdc_binary.c -lm
```

## run the benchmark

```
gcc -std=c99 -O2 -I. -o wine_benchmark examples/wine_benchmark.c hdc.c -lm
./wine_benchmark
```

## warnings

- **call `hdc_init()` before anything else.** level_encode uses a randomized internal table that gets built during init. skip it and you get biased encoding with no error.
- **classifier structs are large.** float is ~5MB, binary is ~5MB (accumulators). declare them `static` or global, never as a local variable inside a function.
  ```c
  static struct hdc_classifier clf;
  hdc_classifier_init(&clf, 4096);
  ```
- **all functions do NULL and bounds checking.** you'll get a printed warning instead of a segfault if you pass bad pointers or invalid dimensions.
- **max dimension is 10048.** configurable via `HDC_MAX_DIMENSION` in the header.
- **circular_convolve requires power-of-2 dimensions** (512, 1024, 2048, 4096, etc) for the fft.
- **binary dimensions must be multiples of 64** since vectors are packed into `uint64_t` words.

## what's next

- gesture recognition demo on pico 2w + mpu6050
- text/language classification via ngram encoding
- simd acceleration (sse2/avx2)
- fpga hdc accelerator prototype

## license

mit
