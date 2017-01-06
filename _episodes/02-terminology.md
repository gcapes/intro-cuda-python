---
title: "Terminology"
teaching: 0
exercises: 0
questions:
- 
- 
objectives:
- "Learn some CUDA terminology"
- "Learn about computer architecture"
keypoints:
- ""
---
## Basic terminology
CUDA is a hybrid programming model, meaning sections of code can execute either on the CPU (host),
or the GPU (device). The term **host** refers to the CPU and its memory; **device** refers to the
GPU and its memory.

A subroutine that executes on the device but is called from the host is called a **kernel**.

FIXME:
So far I have only encountered threads, warps, blocks etc when reading about CUDA C.
