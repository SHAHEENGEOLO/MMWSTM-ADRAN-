# MMWSTM-ADRAN-# MMWSTM-ADRAN+  
*Enhanced Hybrid Deep-Learning Architecture for Climate Time-Series Analysis*

[![Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE) ![PyPI - Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

MMWSTM-ADRAN+ is a **dual-stream neural model for daily climate forecasting and extreme-event detection**.  
It merges a *Multi-Modal Weather State Transition Model* (latent-state bi-LSTM) with an *Anomaly-Driven Recurrent Attention Network*.  
A gated fusion layer delivers forecasts that are simultaneously **regime-aware** and **extreme-sensitive**.

*Trains on 5-year daily records in ≈ 2 h on a single modern GPU (< 5 M parameters).*

---

## Key Features

| Category | Highlights |
|----------|------------|
| **Architecture** | • Latent-state transition matrix inside bi-LSTM<br>• Anomaly-amplified attention branch<br>• Trainable gated fusion |
| **Loss & Metrics** | Extreme-weighted loss boosts tails without sacrificing RMSE; built-in CRPS & extreme-recall metrics |
| **Data Pipeline** | One-click script converts raw CSV/NetCDF → normalised Parquet tensors |
| **Augmentation** | Jitter, time-warp, window-slice, magnitude-warp (factor-3 up-sampling) |
| **Visualisation** | 3-D PCA clusters, attention heat-maps, latent-state probabilities |
| **Reproducibility** | YAML configs, CLI interface, Apache-2.0 licence, Zenodo-archived code & data |

---

## Repository Layout


