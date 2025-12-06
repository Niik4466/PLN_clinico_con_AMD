# Clinical NLP with AMD

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-red?style=for-the-badge&logo=amd&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research-orange?style=for-the-badge)

This repository contains the source code and resources for the development of the paper **"Clinical NLP with AMD"**.

## Paper Purpose

The main objective of this research is to compare the performance of **NVIDIA** and **AMD** graphics cards (GPUs) in the context of Natural Language Processing (NLP) tasks within the clinical domain.

The primary evaluation metric is **FLOPS/Joule**, which allows measuring the energy efficiency of models executed on each hardware architecture.

## Repository Structure

The project is organized as follows:

```
PLN_clinico_con_AMD/
├── GPUMonitor/         # Shared library for GPU monitoring (AMD/NVIDIA)
│   ├── __init__.py
│   └── monitor.py
├── paper_x/            # Experiments and scripts for the x paper
│   ├── AMD/            # Specific code for AMD environment (ROCm)
│   │   ├── Dockerfile
│   │   ├── obtain_data.py
│   │   └── ...
│   └── NVIDIA/         # Specific code for NVIDIA environment (CUDA)
│       ├── Dockerfile
│       ├── obtain_data.py
│       └── ...
├── setup.py            # Installation script for GPUMonitor
└── README.md           # This file
```

## Cloning the Repository

To clone the repository with submodules, use the following command:
```bash
git clone --recurse-submodules https://github.com/Niik4466/PLN_clinico_con_AMD.git
```

## Building Docker Images

To ensure reproducibility and access to the shared `GPUMonitor` library, Docker images must be built from the **project root directory**.

### General Workflow

1.  **Navigate to the project root**:
    ```bash
    cd PLN_clinico_con_AMD
    ```

2.  **Build the image for AMD**:
    ```bash
    docker build -f paper_1/AMD/Dockerfile -t amd-pln-image .
    ```

3.  **Build the image for NVIDIA**:
    ```bash
    docker build -f paper_1/NVIDIA/Dockerfile -t nvidia-pln-image .
    ```

**Important Note**: The dot (`.`) at the end of the command is crucial, as it indicates that the build context is the current directory (the root), allowing Docker to access the `GPUMonitor` folder.
