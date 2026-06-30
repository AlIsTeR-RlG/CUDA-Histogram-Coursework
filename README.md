# CUDA GPU-Accelerated Histogram

A parallel computing project implementing a GPU-accelerated histogram over a large floating-point dataset using CUDA C++. The program computes the minimum and maximum values of the dataset via parallel reduction, then builds a 512-bin histogram — all running on the GPU.

This was completed as coursework for COM2039 (Parallel Computing) at the University of Surrey, exploring GPU memory hierarchies, parallel reduction patterns, and atomic operations. The CUDA implementation in `Q1.cu` is my own work; `main.cpp`, `com2039.hpp`, and the project scaffolding were provided by the module. Note that this repository contains only the implementation portion of the coursework. The full submission also included additional exam-style written questions covering topics such as warp divergence and task graph concurrency, which were answered in the report but have no associated code.

## What it does

1. Loads a binary dataset of 32-bit floats from disk
2. Finds the maximum value using a parallel reduction kernel (`maxReduceKernel`)
3. Finds the minimum value using a parallel reduction kernel (`minReduceKernel`)
4. Computes a 512-bin histogram using a shared-memory-optimised kernel (`histogramKernel512`)
5. Prints each bin count and the total number of elements processed

## Key implementation details

**Parallel reduction** — both the min and max kernels use a tree-based shared memory reduction. Each block reduces its chunk of the input to a single value in shared memory, and the kernel is invoked iteratively until a single value remains across all blocks.

**Shared memory histogram** — rather than having every thread atomically increment the global output histogram directly (which would cause severe memory contention at scale), each block maintains a private histogram in fast on-chip shared memory. Threads update this local copy with `atomicAdd`, and only once per block is the result merged into global memory. This reduces the number of expensive global atomic operations from O(N) to O(grid_size × 512).

**Error handling** — every CUDA API call is wrapped in error checking that prints a descriptive message and exits cleanly on failure.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 11+)
- A C++ compiler compatible with `nvcc`

## Building

```bash
nvcc -o histogram main.cpp Q1.cu
```

## Running

```bash
./histogram path/to/data_points.pbin
```

A sample dataset (`data_points.pbin`) is included in the repository.

Expected output:

```
Read: <N> bytes = <M> elements.
Correctly loaded data_points.pbin
length of vector <M>
GPU Max: <value>
GPU Min: <value>
Bin[0]: <count>
Bin[1]: <count>
...
Bin[511]: <count>
Total number of elements in histogram: <M>
```

## Project structure

| File | Description |
|------|-------------|
| `Q1.cu` | CUDA kernels and host wrapper functions (min, max, histogram) — **written by me** |
| `main.cpp` | Entry point: loads data, calls kernels, prints results *(provided by module)* |
| `com2039.hpp` | Shared header: kernel declarations and constants *(provided by module)* |
| `data_points.pbin` | Binary input dataset |
| `report_6849160.pdf` | Written report covering design decisions and profiling analysis.|
| `prof_out6849160.pfout` | Nsight profiler output |

## Configuration

Two constants in `com2039.hpp` control the kernel parameters:

```cpp
const size_t BLOCK_SIZE = 1024;  // Threads per block
const size_t NUM_BINS   = 512;   // Histogram bins
```
