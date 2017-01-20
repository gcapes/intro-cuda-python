---
title: "Anja's stuff"
teaching: 0
exercises: 0
questions:
-
objectives:
-
keypoints:
- 
- 
---
## Intro/Definition
[Wikipedia.org](https://en.wikipedia.org/wiki/CUDA)
**CUDA** is a parallel computing platform and application programming interface (API) model created by Nvidia.[1] It allows software developers and software engineers to use a *CUDA-enabled graphics processing unit (GPU)* for *general purpose* processing – an approach termed GPGPU (General-Purpose computing on Graphics Processing Units). The CUDA platform is a *software layer* that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of compute kernels.
The CUDA platform is designed to work with programming languages such as C, C++, and Fortran. CUDA supports programming frameworks such as OpenACC and OpenCL. Third party wrappers are also available for Python, Perl, Fortran, Java, Ruby, Lua, Haskell, R, MATLAB, IDL, and native support in Mathematica.
Used for:
* game physics calculations/pysics engines
* computational biology
* cryptography

CUDA provides both *low-level* API and *higher-level* API
CUDA is compatible with most standard operating systems. 

## Glossery
First resource
[Warps and occupancy](http://on-demand.gputechconf.com/gtc-express/2011/presentations/cuda_webinars_WarpsAndOccupancy.pdf)

* A grid is composed of blocks which are completely independent
* A block is composed of threads which can communicate within their own block
* 32 threads form a warp Instructions are issued per warp
* If an operand is not ready the warp will stall
  * Context switch between warps when stalled
  * Context switch must be very fast
* Occupancy = Active Warps / Maximum Active Warps

## Components
CUDA 8.0 comes with the following libraries:
* CUBLAS - CUDA Basic Linear Algebra Subroutines library
* CUDART - CUDA RunTime library
* CUFFT - CUDA Fast Fourier Transform library
* CURAND - CUDA Random Number Generation library
* CUSOLVER - CUDA based collection of dense and sparse direct solvers
* CUSPARSE - CUDA Sparse Matrix library
* NPP - NVIDIA Performance Primitives library
* NVGRAPH - NVIDIA Graph Analytics library
* NVML - NVIDIA Management Library
* NVRTC - NVRTC RunTime Compilation library for CUDA C++

CUDA 8.0 comes with these other software components:
* nView - NVIDIA nView Desktop Management Software
* NVWMI - NVIDIA Enterprise Management Toolkit
* PhysX - GameWorks PhysX is a scalable multi-platform game physics solution
## Advantages:
* Scattered reads – code can read from arbitrary addresses in memory
* Unified virtual memory (CUDA 4.0 and above)
* Unified memory (CUDA 6.0 and above)
* Shared memory – CUDA exposes a fast shared memory region that can be shared among threads. This can be used as a user-managed cache, enabling higher bandwidth than is possible using texture lookups.
* Faster downloads and readbacks to and from the GPU
* Full support for integer and bitwise operations, including integer texture lookups
