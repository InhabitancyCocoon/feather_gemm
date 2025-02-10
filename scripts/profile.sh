#!/bin/bash

# Set directories

DEVICE="A100-PCIE-40GB"

OUTPUT_DIR="docs/profile_result/${DEVICE}"
EXECUTABLE_DIR="build"

# NVreg_RestrictProfilingToAdminUsers=0

# Run the ncu command
ncu --set full -o "${OUTPUT_DIR}/00_cublas"              "${EXECUTABLE_DIR}/main" 0
ncu --set full -o "${OUTPUT_DIR}/01_naive"               "${EXECUTABLE_DIR}/main" 1
ncu --set full -o "${OUTPUT_DIR}/02_coalesce_memory"     "${EXECUTABLE_DIR}/main" 2
ncu --set full -o "${OUTPUT_DIR}/03_tiled"               "${EXECUTABLE_DIR}/main" 3
ncu --set full -o "${OUTPUT_DIR}/04_4x_coarsen_tiled"    "${EXECUTABLE_DIR}/main" 4
ncu --set full -o "${OUTPUT_DIR}/05_8x_compute"          "${EXECUTABLE_DIR}/main" 5
ncu --set full -o "${OUTPUT_DIR}/06_more_compute"        "${EXECUTABLE_DIR}/main" 6
ncu --set full -o "${OUTPUT_DIR}/07_vectorize"           "${EXECUTABLE_DIR}/main" 7
ncu --set full -o "${OUTPUT_DIR}/08_warp_tile"           "${EXECUTABLE_DIR}/main" 8
ncu --set full -o "${OUTPUT_DIR}/09_double_buffering"           "${EXECUTABLE_DIR}/main" 9