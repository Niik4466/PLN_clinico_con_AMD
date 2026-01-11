FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

COPY paper_1/NVIDIA/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY paper_1/NVIDIA/download_dataset.py /tmp/download_dataset.py
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/Dataset/
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/OutPut

RUN python3 /tmp/download_dataset.py

RUN rm -rf /home/data3/Ali/Code/Saina/Brea/Dataset/IDC_regular_ps50_idx5

# --- Install GPUMonitor ---
# Assumes build context is project root
COPY GPUMonitor /GPUMonitor
COPY setup.py /setup.py
RUN pip install -e /

COPY paper_1/NVIDIA /Breast-Cancer-Classification/

WORKDIR /Breast-Cancer-Classification
