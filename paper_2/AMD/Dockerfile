# Dockerfile AMD - Paper 2: Clinical NLP ICD-10 Classification
# Build from project root: docker build -f paper_2/AMD/Dockerfile -t paper2-amd .

FROM rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.9_pytorch_release_2.4.0

# 1. Install dependencies
COPY paper_2/AMD/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2. Install system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libc6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install GPUMonitor
COPY GPUMonitor /GPUMonitor
COPY setup.py /setup.py
RUN pip install -e /

# 4. Create project directories
RUN mkdir -p /app/data /app/models

# 5. Copy scripts and datasets
COPY paper_2/AMD/train_top10_models.py /app/train_top10_models.py
COPY paper_2/AMD/train_top50_models.py /app/train_top50_models.py
COPY paper_2/AMD/data/df_final_top10.csv /app/data/df_final_top10.csv
COPY paper_2/AMD/data/df_final_top50.csv /app/data/df_final_top50.csv

# 6. Set working directory
WORKDIR /app

# 7. ROCm environment variable
ENV HSA_FORCE_FINE_GRAIN_PCIE=1

# 8. Default command
CMD ["python3", "train_top10_models.py"]
