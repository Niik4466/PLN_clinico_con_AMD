FROM rocm/pytorch:rocm6.3.3_ubuntu20.04_py3.9_pytorch_release_2.2.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY download_dataset.py /tmp/download_dataset.py
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/Dataset/
RUN mkdir -p /home/data3/Ali/Code/Saina/Brea/Output

RUN python3 /tmp/download_dataset.py

RUN rm -rf /home/data3/Ali/code/Saina/Brea/Dataset/IDC_regular_ps50_idx5

COPY Breast-Cancer-Classification/ /Breast-Cancer-Classification/

WORKDIR /Breast-Cancer-Classification