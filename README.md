# ECC CUDA Project

This project implements an error correction (ECC) algorithm combined with a Rodinia benchamrks on a GPU using CUDA.

## Prerequisites

- CUDA Toolkit (with `nvcc`)
- NVIDIA Nsight Systems (`nsys`)
- Rodinia Benchmark Suite (specifically the BFS and Gaussian Elimination dataset)
- An NVIDIA GPU supporting architecture `sm_75` or higher

## Setting up NVIDIA MPS

Enable NVIDIA Multi-Process Service (MPS). MPS allows multiple CUDA applications to run concurrently on the same GPU.

1. Start the MPS control daemon:

    ```bash
    nvidia-cuda-mps-control -d
    ```

2. Set the pipe and log directories to temporary locations:

    ```bash
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    ```

3. Restrict your application to a specific GPU (e.g., GPU 0):

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    ```

After completing these steps, MPS will be enabled and ready for use with your CUDA app

## Compilation

To compile the CUDA files for this project, use the following command.

```bash
nvcc -o ecc_bfs ecc.cu ecc_main.cu bfs_new.cu kernel.cu kernel2.cu -arch=sm_75
```

## Profiling

```bash
nsys profile -o report ./ecc_bfs ../rodinia_3.1/data/bfs/graph1MW_6.txt &
./bfs ../rodinia_3.1/data/bfs/graph1MW_6.txt &
```

Utilize the same flow for Gaussian Elimination
