# GAT for DOGM

This repository contains an implementation of a Graph Attention Network (GAT) for Dynamic Occupancy Grid Maps (DOGM).

## Environment Setup

This project was developed and tested in the following environment:

*   **Python:** `3.11.8`
*   **PyTorch:** `2.5.1+cu121`
*   **GPU:** `NVIDIA GeForce RTX 4060 Ti`
*   **NVIDIA Driver Version:** `560.94`
*   **CUDA Version (from driver):** `12.6`

### Notes
The PyTorch build is linked against CUDA 12.1 (`+cu121`), while the system's NVIDIA driver supports up to CUDA 12.6. This is a compatible configuration.
