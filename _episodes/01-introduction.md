---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is CUDA"
- "What is a GPU"
objectives:
- "Learn what CUDA is used for"
- "Learn about computer architecture"
keypoints:
- "CUDA is used for parallel programming on Nvidia GPU cards"
---
## A brief history of GPU computing

### Multi-core central processing units (CPUs)
Almost all consumer PCs and laptops have central processing units (CPUs) with multiple cores.
Many mobile phones and tablets have multiple cores too.
The reason for multiple cores is that manufacturers have been unable to increase performance
by increasing clock speed (clock speed measures the rate at which calculations can be performed,
i.e. how quickly a CPU is able to execute instructions).

### Graphics processing units (GPUs)
The level of parallelism in CPUs (a few -- tens of cores) is insignificant compared to the
parallelism found in GPUs (typically 2000-3000 cores).
Graphics cards are massively parallel processors, optimised to do floating point arithmetic.
This has been driven by the 3D graphics requirements of the games industry, but this also happens
to be just what is required for most computationally expensive scientific problems 
(linear algebra matrix operations).

## What is CUDA?
**C**ompute **U**nified **D**evice **A**rchitecture, 
a parallel computing platform and application programming interface (API), created by Nvidia.

CUDA can be used to leverage the parallel processing capabilities of Nvidia GPUs.
There are various interfaces to this, including CUDA-accelerated libraries, compiler directives,
and extensions to programming languages e.g. CUDA C, which is essentialy the C programming
language with a few extensions that can run sections of code on the GPU.

Third party wrappers are available for Python.

CUDA programming is a hybrid model where serial sections of code run on the CPU and parallel
sections run on the GPU.

### Types of parallelism
The hybrid nature of CUDA means that computation on the CPU can overlap with computation on
the GPU.

Far greater parallelism occurs within the GPU itself, where subroutines are executed by many
threads in parallel.
These threads execute the same code, but on different data.
For example adjacent threads may operate on adjacent data such as elements of an array.
Contrast with a model like Message Passing Interface (MPI) where the data is split into portions
and each MPI process performs calculations on an entire data segment.
