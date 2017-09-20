---
title: "Examples"
teaching: 0
exercises: 0
questions:
- "What type of operations can be accelerated using libraries"
- "How can libraries be used to accelerate calculations"
- "How can CUDA python be used to write my own kernels"
- "Worked examples moving from division between vectors to sum reduction"
objectives:
- "Learn to use CUDA libraries"
- "Learn to accelerate Python code using CUDA"
keypoints:
- 
- 
---
Show examples for each of the CUDA use scenarios mentioned:

- libraries
- compiler directives - not applicable to python?
- programming languages: CUDA Python

After visiting a great number of web pages this week,
this [NVidia page](https://developer.nvidia.com/how-to-cuda-python) is the main source I have
settled on.

There are two examples here using Anaconda NumbaPro. 

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

Thankfully the [Numba documentation](http://numba.pydata.org/numba-doc/0.30.0/index.html)
looks fairly comprehensive and includes some examples.

## PyCUDA
Looks to be just a wrapper to enable calling kernels written in CUDA C.
This would seem to be out of the scope of this course?

FIXME:
Find some examples for some of the above (more on GPU obviously).
Some material [here](https://developer.nvidia.com/cuda-education), the most useful being examples
on github:

[Continuum Analytics NumbaPro repo](https://github.com/ContinuumIO/numbapro-examples)

[NVIDIA NumbaPro repo](https://github.com/harrism/numbapro_examples)

## To do list for understanding:

- [NVidia page examples](https://developer.nvidia.com/how-to-cuda-python)
(See [code]({{ page.root }}/code) folder)
	- Mandlebrot example
		- Get last section "Even Bigger Speedups with CUDA Python" working.

		I have tried the Mandlebrot example on Zrek, and only the first part works.
		I have emailed Nvidia and the GitHub repo owner asking for help updating
		this code which uses Numbapro (deprecated).
		~~No reponse received. **Help required!**~~
		I just got a [response](https://github.com/harrism/numba_examples/issues/2#issuecomment-330739425), suggesting this has now been fixed.

		- The [github](https://github.com/ContinuumIO/numbapro-examples/tree/master/mandel)
		repo has a different sequence of steps, so look at what has been
		done there as well.
	- ~~Try the other (Monte Carlo) example from [same page](https://developer.nvidia.com/how-to-cuda-python)~~
		- Understand how the speed ups work in the monte-carlo example
		then document it

- ~~Read Numba user manual~~
- Read Numba documentation
- Find library examples using anaconda accelerate e.g. cuBLAS
- Find MKL examples using anaconda accelerate
- Work through [this set](https://github.com/ContinuumIO/intel_hpc_2016_numba_tutorial) 
of jupyter notebooks, which looks to be a sub-set of 
[this](https://github.com/ContinuumIO/supercomputing2016-python) python resource
	- Look out particularly for @vectorize
- Read [CUDA C programming guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4WJRtDMGF)
for the detail of how CUDA works

## To do list for lesson structure:
- Explain what different Numba options are used for:
	- numba.jit: CPU compilation of python code. 'cache' option for quicker subsequent calls
	- numba.vectorize: ufunc with scalar input. Target options: cpu, parallel, cuda
	- numba.guvectorize: as above but with input is an arbitrary number of array elements
	- For all three above, 'nopython' option is quicker
- What to compile: identify critical paths in code (ok, but which profiler?)
- Link to [Numba troubleshooting page](http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html)
