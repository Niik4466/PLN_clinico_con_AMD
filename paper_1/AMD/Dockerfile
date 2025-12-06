FROM rocm/pytorch:rocm6.3.3_ubuntu20.04_py3.9_pytorch_release_2.2.1

COPY paper_1/AMD/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN apt update && apt install libc6 -y

COPY paper_1/AMD/download_dataset.py /tmp/download_dataset.py
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/Dataset/
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/OutPut

RUN python3 /tmp/download_dataset.py

RUN rm -rf /home/data3/Ali/Code/Saina/Brea/Dataset/IDC_regular_ps50_idx5

# --- Install GPUMonitor ---
# Assumes build context is project root
COPY GPUMonitor /GPUMonitor
COPY setup.py /setup.py
RUN pip install -e /

COPY paper_1/AMD /Breast-Cancer-Classification/

WORKDIR /Breast-Cancer-Classification
