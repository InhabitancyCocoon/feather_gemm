#pragma once

#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_coalecse_memory.cuh"
#include "kernels/03_tiled.cuh"
#include "kernels/04_4x_coarsen_tiled.cuh"
#include "kernels/05_8x_compute.cuh"
#include "kernels/06_more_compute.cuh"
#include "kernels/07_vectorize.h"
#include "kernels/08_warp_tile.cuh"