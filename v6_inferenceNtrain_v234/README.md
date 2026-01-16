# k_nn_temporal training session

This folder contains the results of a training and visualization session.

## Settings

The following settings were used for this session, extracted from the `ModelConfig` in `gat_train_stgat_dogm.py`:

### Model Configuration
- **INPUT_DIM**: 11
- **IDX_POS**: (0, 1)
- **IDX_DT**: 2
- **IDX_EGO**: (3, 4)
- **IDX_INTENSITY**: 5
- **IDX_VR**: 6
- **IDX_SNR**: 7
- **IDX_SID**: (8, 9, 10)

### Window / Frames
- **WINDOW**: 4
- **DT_DEFAULT**: 0.1

### Downsample Caps (Training)
- **LIDAR_CAP_PER_FRAME**: 512
- **RADAR_CAP_PER_FRAME**: 128

### Graph Construction
- **K_LIDAR_SPATIAL**: 32
- **K_RADAR_SPATIAL**: 8
- **K_TEMPORAL**: 8
- **TEMPORAL_ADJ_ONLY**: True

### Cross Edges
- **CROSS_RADIUS**: 0.5
- **K_CROSS_L2R**: 4
- **K_CROSS_R2L**: 8
- **CROSS_DROPOUT**: 0.2

### Model
- **HIDDEN_DIM**: 64
- **NUM_HEADS**: 4
- **DROPOUT**: 0.1
- **EDGE_DIM**: 4

### Sigma Constraints
- **MIN_LOG_SIGMA**: -5.0
- **MAX_LOG_SIGMA**: 2.0

### Loss
- **REG_LAMBDA**: 1e-3
- **RADAR_LOSS_WEIGHT**: 5.0
- **ASSOC_TOPK**: 5
- **ASSOC_TAU**: 0.5
- **ASSOC_GATE_LIDAR**: 0.15
- **ASSOC_GATE_RADAR**: 0.3

### Cross Residual Schedule
- **CROSS_ALPHA_MAX**: 0.3
- **CROSS_WARMUP_STEPS**: 2000
- **CROSS_RAMP_STEPS**: 8000

### Training Defaults
- **BATCH_SIZE**: 2
- **LR**: 3e-4
- **WEIGHT_DECAY**: 1e-4
- **EPOCHS**: 30
- **GRAD_CLIP**: 1.0
- **AMP**: False
