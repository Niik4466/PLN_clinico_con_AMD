# Dockerfile NVIDIA - Paper 2: Clinical NLP ICD-10 Classification
# Build from project root: docker build -f paper_2/NVIDIA/Dockerfile -t paper2-nvidia .

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 1. Install dependencies
COPY paper_2/NVIDIA/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2. Install GPUMonitor
COPY GPUMonitor /GPUMonitor
COPY setup.py /setup.py
RUN pip install -e /

# 3. Create project directories
RUN mkdir -p /app/data /app/models

# 4. Copy scripts and datasets
COPY paper_2/NVIDIA/train_top10_models.py /app/train_top10_models.py
COPY paper_2/NVIDIA/train_top50_models.py /app/train_top50_models.py
COPY paper_2/NVIDIA/data/df_final_top10.csv /app/data/df_final_top10.csv
COPY paper_2/NVIDIA/data/df_final_top50.csv /app/data/df_final_top50.csv

# 5. Set working directory
WORKDIR /app

# 6. Default command
CMD ["python3", "train_top10_models.py"]
