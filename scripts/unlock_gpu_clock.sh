#!/bin/bash

nvidia-smi --reset-gpu-clocks
nvidia-smi --reset-memory-clocks
nvidia-smi --persistence-mode=0