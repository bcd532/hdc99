# hdc99

![C99](https://img.shields.io/badge/C99-00599C?style=flat&logo=c&logoColor=white)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-0.3.2-orange)
![Platform](https://img.shields.io/badge/platform-any-lightgrey)

Standalone hyperdimensional computing library in C99. Zero dependencies, ~15KB of source across two files, runs anywhere.

Float vectors and binary (bit-packed) vectors. FFT-based circular convolution. Drop-in ready.

## what is this?

HDC encodes data into high-dimensional vectors and classifies by similarity — no training loop, no backpropagation, no GPU. It works on microcontrollers, laptops, and bare metal.

Its niche is **footprint, not peak accuracy**: the whole library is a couple of small C files with no dependencies, and on standard benchmarks it lands competitive-with-but-not-ahead-of classical baselines while being small enough to run on a Pico. If you need the highest possible accuracy on a workstation, a tuned SVM will beat it (see below). If you need decent classification in 15KB with no dependencies, that's the trade this is built for.

## benchmarks

All numbers below are reproducible with the commands shown. They are single-machine results on UCI Wine (178 samples, 13 features, 3 classes), 80/20 train/test split, min-max scaling fit on the training split only. Accuracy varies seed-to-seed, so results are reported as a distribution over many random splits, not a single run.

### wine accuracy (float HDC, DIM 512, 200 random splits)

| statistic | value  |
| --------- | ------ |
| mean      | 96.3%  |
| std       | ± 2.9  |
| min       | 86.1%  |
| median    | 97.2%  |
| max       | 100.0% |

100% accuracy occurs on ~22% of random splits (43 / 200). A single run can therefore print anything from ~86% to 100%; the mean is the honest headline number.

### baselines on the same data and protocol (200 splits, scikit-learn)

| model         | mean accuracy |
| ------------- | ------------- |
| SVM (rbf)     | 98.6%         |
| SVM (linear)  | 98.3%         |
| GaussianNB    | 97.6%         |
| **hdc99**     | **96.3%**     |
| kNN (k=3)     | 96.0%         |
| kNN (k=5)     | 95.7%         |

Read this honestly: hdc99 is **statistically tied with kNN** and **trails SVM by ~2 points** (SVM is also more consistent). The value proposition is not that it wins the accuracy table — it doesn't. It's that it reaches this accuracy in ~15KB of dependency-free C.

### dimension scaling (float HDC, 100 random splits per size)

| dimension | mean   | std   |
| --------- | ------ | ----- |
| 32        | 87.1%  | ± 6.4 |
| 64        | 92.4%  | ± 4.9 |
| 128       | 94.3%  | ± 3.2 |
| 256       | 95.3%  | ± 3.4 |
| 512       | 96.4%  | ± 3.0 |
| 1024      | 96.2%  | ± 2.8 |
| 2048      | 96.6%  | ± 2.6 |
| 4096      | 96.4%  | ± 2.7 |

Accuracy climbs to ~512 dimensions and then **saturates** — going higher costs memory for no measurable accuracy gain. Dropping to 64 dimensions is possible but costs ~4 points of accuracy and roughly doubles run-to-run variance, so treat 64 as a memory-vs-accuracy trade, not a free lunch. ~512 is the practical sweet spot for this dataset.

## reproduce the benchmarks

Single split (prints one accuracy — note this will vary run to run):

```
gcc -std=c99 -O2 -I. -o wine_benchmark examples/wine_benchmark.c hdc.c -lm
./wine_benchmark
```

The multi-seed distribution and dimension sweep used to produce the tables above are straightforward loops over the single-split harness, varying the RNG seed for both the split (`srand`) and the hypervectors (`hdc_init`). See `examples/` for the harnesses.

## what you get

### float (hdc.h + hdc.c)

**primitives**

- `bind` — element-wise multiply, encodes relationships, reversible
- `bundle` — element-wise add, combines vectors like a vote
- `permute` — circular shift, encodes position and order
- `normalize` — scale to unit length for fair comparisons
- `similize` — cosine similarity between two vectors
- `random_bipolar` — generate random vectors of -1 and 1
- `circular_convolve` — FFT-based circular convolution (HRR-style binding)

**encoding**

- `level_encode` — continuous value (0.0–1.0) to vector, randomized flip order for zero bias
- `id_level_encode` — multi-channel sensor data to one vector with channel identity
- `ngram` — sequence fingerprinting for pattern and order capture

**classification**

- `train` — add examples to class prototypes
- `classify` — find the most similar class, returns -1 if nothing trained
- `hdc_classifier_init` — initialize classifier with dimension

**fft**

- `fft` / `inverse_fft` — fast Fourier transform on complex arrays
- `circular_convolve` — convolve two vectors via FFT
- `complex_multiply` — complex number multiplication
- `vector_to_complex` / `complex_to_vector` — conversion helpers

**helpers** — `zero_vector`, `neg_vector`, `copy_vector`, `shuffle`, `check_null`, `check_dimension`

### binary (hdc_binary.h + hdc_binary.c)

Bit-packed vectors in `uint64_t` arrays — 64 dimensions per word. Faster and lighter than float.

- `random_binary` — generate random bit vectors
- `bind_binary` — XOR binding (single CPU instruction per 64 dims)
- `bundle_binary` — majority vote across multiple vectors
- `similize_binary` — Hamming distance via popcount
- `permute_binary` — bit-level circular shift
- `level_encode_binary` — continuous value to binary vector
- `id_level_encode_binary` — multi-channel sensor encoding
- `train_binary` / `classify_binary` — accumulator-based classifier with majority-vote thresholding
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

For binary HDC:

```
gcc -std=c99 -I. -o app your_file.c hdc_binary.c -lm
```

## warnings

- **call `hdc_init()` before anything else.** `level_encode` uses a randomized internal table built during init. Skip it and you get biased encoding with no error.
- **classifier structs are large** (~5MB). Declare them `static` or global, never as a local variable inside a function.
- **all functions do NULL and bounds checking** — you get a printed warning instead of a segfault on bad pointers or invalid dimensions.
- **max dimension is 10048**, configurable via `HDC_MAX_DIMENSION` in the header.
- **`circular_convolve` requires power-of-2 dimensions** (512, 1024, 2048, 4096, …) for the FFT.
- **binary dimensions must be multiples of 64** since vectors are packed into `uint64_t` words.

## what's next

- gesture recognition demo on Pico 2W + MPU6050
- text/language classification via ngram encoding
- SIMD acceleration (SSE2/AVX2)
- FPGA HDC accelerator prototype

## license

MIT
