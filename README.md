# DS2-HLS

Code repository for the paper:  
**"Two-Stage Distribution-Aware Learning and Gradient Conflict Mitigation for Robust HLS Design Prediction"**

---

## 📌 Overview

This repository provides the official implementation of the **DS2-HLS** framework for robust performance and resource prediction in High-Level Synthesis (HLS) design. Our method incorporates:

- **Two-Stage Distribution-Aware Learning**:
  - Stage I: Learn inter-kernel distribution with Intra-Kernel Instance Distance Loss (IIKDL)
  - Stage II: Residual-based fine-tuning using a Residual Fitting Module (RFM)
- **Multi-Task Gradient Conflict Mitigation** using MA-PCGrad

---

## 🛠️ Installation

We recommend creating a virtual environment using `conda`:

```bash
conda create --name ds2 python==3.12.2
conda activate ds2
pip install -r requirements.txt
```

## 📁 Directory Structure

```
├── dataset
│   ├── designs
│   ├── encoders.klepto
│   ├── graphs
│   ├── pragma_dim.klepto
│   └── sources
├── README.md
├── requirements.txt
└── src
    ├── config_ds.py
    ├── config.py               # Configuration file
    ├── data.py                 # Data gen file
    ├── dse.py                  # Design Space Exploration
    ├── main.py                 # Entry point for training
    ├── mapcgrad.py
    ├── nn_att.py
    ├── parameter.py
    ├── __pycache__
    ├── result.py
    ├── saver.py
    ├── solver                  # Models and Train
    └── utils.py                # Utility functions
```

## 🚀 Usage Guide
1. Configuration
    All training-related options are controlled via src/config.py.
    To regenerate the dataset during training, set:

    ```python
    force_regen = True
    ```
    If using IIKDL, enable:

    ```python
    enable_ikdl = True
    ```

    To run Stage II (boost) training, you must:

    First complete Stage I training and obtain the base model

    Then, set the following in config.py:

    ```python
    enable_boost = True
    boost_base_model_path = '<path_to_stage1_model>'
    boost_use_mtl = True  # Set False if using base instead of mtl
    ```

2. Training
    Training scripts are organized under the solver/ directory. To train a model, ensure that the correct imports are used in main.py. For example, to train the MTL model:

    ```python
    from solver.mtl.train import train_main, inference
    from solver.mtl.model import Net
    ```

    Then run:

    ```bash
    python src/main.py
    ```

## 📝 Notes
If this is your first run, be sure to regenerate the dataset with force_regen = True

The Stage II model (boost) requires Stage I outputs

All parameters should be configured in config.py before training