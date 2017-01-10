---
title: "Examples"
teaching: 0
exercises: 0
questions:
- 
- 
objectives:
- "Learn to use CUDA libraries"
- "Learn to accelerate Python code using CUDA"
keypoints:
- ""
---
Show examples for each of the CUDA use scenarios mentioned:

- libraries
- compiler directives - not applicable to python?
- programming languages: CUDA Python

After visiting a great number of web pages this week,
this [NVidia page](https://developer.nvidia.com/how-to-cuda-python) is the main source I have
settled on.

There are two examples here using Anaconda Accelerate. 
Try these on Zrek.

There is lots of documentation to read on the Continuum Analytics website - linked to at the above site


## Libraries
[Anaconda accelerate](https://docs.continuum.io/accelerate/cudalibs)
provides access to numerical libraries optimised for performance on **Intel CPUs** and **NVidia GPUs**.
Using accelerate, you can access

* CUDA library functions for 
    * basic linear algebra (BLAS)
    * sparse matrices
PyCUDA allows you to call kernels written in CUDA C - so not appropriate for this course?
    * Fast Fourier Tranforms (FFT)
    * random numbers
    * sorting
* Intel Math Kernal Library (MKL) functions for faster BLAS, core maths (UFunc) and FFT operations 
using the CPU in
    * NumPy
    * SciPy
    * scikit-learn
    * NumExpr
* a profiler (so you know well you are doing) 

## Compiler directives
I read about **@vectorize** for automatically accelerating functions, but everything pointed to NumbaPro 
which has been depreciated.
This 
[blog post](https://www.continuum.io/blog/developer-blog/deprecating-numbapro-new-state-accelerate-anaconda)
indicates what has gone where (NumbaPro was paid-for software: now split into Numba (open-source) and 
Accelerate (free for academic use).

[Some Numba examples](http://numba.pydata.org/numba-doc/dev/user/examples.html)

[Numba user manual](http://numba.pydata.org/numba-doc/latest/user/index.html)

## CUDA Python
CUDA functionality can accessed directly from Python code. Information on 
[this page](https://docs.continuum.io/numbapro/CUDAJit) 
is a bit sparse.

Thankfully the Numba [documentation](http://numba.pydata.org/numba-doc/0.30.0/index.html)
looks fairly comprehensive and includes some examples.

## PyCUDA
Looks to be just a wrapper to enable calling kernels written in CUDA C.
This would seem to be out of the scope of this course?

FIXME:
Find some examples for some of the above (more on GPU obviously).
Some material [here](https://developer.nvidia.com/cuda-education), the most useful being examples
on github:

[Continuum Analytics repo](https://github.com/ContinuumIO/numbapro-examples)

[NVIDIA repo](https://github.com/harrism/numbapro_examples)
