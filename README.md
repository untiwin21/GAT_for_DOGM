# GAT for DOGM

This repository contains a PyTorch implementation of a Spatio-Temporal Graph Attention Network (GAT) for learning LiDAR and Radar measurement uncertainty for Dynamic Occupancy Grid Maps (DOGM).

## Overview

The core of this project is the `gat_train_stgat_dogm.py` script, which provides functionalities for:

1.  **Training**: Learns per-point uncertainty for LiDAR position (sigma_x, sigma_y) and Radar radial-velocity (sigma_v).
2.  **Inference**: Runs the trained network on all points and exports the per-frame sigma values.
3.  **Diagnostics**: Provides tools to analyze the model's output and calibration.

## Setup

### Dependencies

This project requires Python 3 and several libraries. You can install them using pip:

```bash
pip install numpy torch torch_cluster torch_scatter matplotlib
```

Note: `torch_cluster` and `torch_scatter` might require a specific PyTorch and CUDA version. Please refer to their official installation instructions.

### Data

The model expects the data to be in a specific format, with files for LiDAR, Radar, and odometry for each run version. The project should be structured as follows:

```
.
├── 20251231_dataset/
│   ├── LiDARMap_BaseScan_v...
│   ├── Radar1Map_BaseScan_v...
│   ├── Radar2Map_BaseScan_v...
│   └── odom_filtered_v...
├── dataset/
│   ├── LiDARMap_BaseScan_v...
│   ├── Radar1Map_BaseScan_v...
│   ├── Radar2Map_BaseScan_v...
│   └── odom_filtered_v...
├── gat_train_stgat_dogm.py
├── sigma_diagnostics.py
└── visualize_v6.py
```

## Usage

### Training

There are two modes for training:

*   **Debug Mode (v1 only)**: For quickly testing the training pipeline on a smaller dataset.
    ```bash
    python gat_train_stgat_dogm.py --task train --data_root dataset --mode debug --batch 2 --num_workers 4 --max_windows_per_run 800
    ```

*   **Full Training (v1-v3 train, v4 val)**: To train the model on the full dataset and validate on a separate set. This will produce `best_ckpt.pt`.
    ```bash
    python gat_train_stgat_dogm.py --task train --data_root dataset --mode train --batch 2 --num_workers 4
    ```

### Inference

To run inference on a specific data version and export the results:

```bash
python gat_train_stgat_dogm.py --task infer --data_root 20251231_dataset --ckpt best_ckpt.pt --infer_version 6 --infer_out sigma_v6.npz --infer_full_points
```

### Diagnostics

To run diagnostics on the model's output:

```bash
python sigma_diagnostics.py --data_root dataset --ckpt best_ckpt.pt --version 4 --batch 2 --num_workers 4
```

### Visualization

To visualize the output of an inference run:

```bash
python visualize_v6.py
```
This will generate a `v6_visualization.png` file in the project root.
