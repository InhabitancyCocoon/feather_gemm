#!/bin/bash

nvidia-smi --persistence-mode=1

# NVIDIA A100-PCIE-40GB
nvidia-smi --lock-gpu-clocks=765
nvidia-smi --lock-memory-clocks=1215