FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY download_dataset.py /tmp/download_dataset.py
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/Dataset/
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/Output

RUN python3 /tmp/download_dataset.py

RUN rm -rf /home/data3/Ali/code/Saina/Brea/Dataset/IDC_regular_ps50_idx5

COPY Breast-Cancer-Classification/ /Breast-Cancer-Classification/

WORKDIR /Breast-Cancer-Classification