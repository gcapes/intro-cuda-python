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
First resource
[Warps and occupancy](http://on-demand.gputechconf.com/gtc-express/2011/presentations/cuda_webinars_WarpsAndOccupancy.pdf)

* A grid is composed of blocks which are completely independent
* A block is composed of threads which can communicate within their own block
* 32 threads form a warp Instructions are issued per warp
* If an operand is not ready the warp will stall
  * Context switch between warps when stalled
  * Context switch must be very fast
* Occupancy = Active Warps / Maximum Active Warps
