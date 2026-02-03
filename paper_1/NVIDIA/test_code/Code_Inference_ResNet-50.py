# Medicion de metricas
from obtain_data import GpuMonitor, export_model_info

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import logging
import torch
import time
from statistics import mean

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the dataset class
class IDCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Set the dataset directory
dataset_dir = '/home/data3/Ali/Code/Saina/Brea/Dataset/'
# dataset_dir = '/home/data3/Ali/Code/Saina/Brea/TestData/'

# Collect image paths and labels
logger.info('Collecting image paths and labels...')
image_paths = []
labels = []

for folder_name in os.listdir(dataset_dir):
    class_dir_0 = os.path.join(dataset_dir, folder_name, '0')
    class_dir_1 = os.path.join(dataset_dir, folder_name, '1')
    for img_name in os.listdir(class_dir_0):
        image_paths.append(os.path.join(class_dir_0, img_name))
        labels.append(0)
    for img_name in os.listdir(class_dir_1):
        image_paths.append(os.path.join(class_dir_1, img_name))
        labels.append(1)

logger.info(f'Collected {len(image_paths)} images.')

# Split the data into training, validation, and testing sets
logger.info('Splitting data into training, validation, and testing sets...')
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.3333, stratify=temp_labels, random_state=42)

logger.info(f'Training set size: {len(train_paths)}')
logger.info(f'Validation set size: {len(val_paths)}')
logger.info(f'Test set size: {len(test_paths)}')

# Data transforms without augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets
logger.info('Creating datasets...')
train_dataset = IDCDataset(train_paths, train_labels, transform=data_transforms['train'])
val_dataset = IDCDataset(val_paths, val_labels, transform=data_transforms['val'])
test_dataset = IDCDataset(test_paths, test_labels, transform=data_transforms['test'])

# Create dataloaders
logger.info('Creating dataloaders...')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pretrained ResNet-50 model
logger.info('Loading pretrained ResNet-50 model...')
model = models.resnet50(pretrained=True)

# Modify the final layer for binary classification with dropout
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1)
)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------ EXPORTAR INFO DEL MODELO ---------------------
export_model_info(model=model, output_dir='Data/model_info.csv')
# -----------------------------------------------------------------------------

def medir_eficiencia(model, monitor, num_iters=100, batch_size=32, input_shape=(3, 224, 224)):
    model.eval()
    device = next(model.parameters()).device
    
    # Construir input con dimension de batch
    # Nota: input_shape debe ser (C, H, W)
    full_input_shape = (batch_size, *input_shape)
    dummy_input = torch.randn(full_input_shape, device=device)

    # iniciar monitor
    monitor.clear()
    monitor.start()

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in tqdm(range(num_iters), desc="Ejecutando iteraciones"):
            _ = model(dummy_input)
            torch.cuda.synchronize()  # asegurar que se complete cada 

    total_time = time.time() - start_time
    monitor.stop()

    monitor.export_to_csv(filename_prefix="Inference_ResNet50_NVIDIA_raw", output_path="Data")

    # obtener estadísticas de GPU
    stats = monitor.get_stats()
    power_avg_w = stats.get("gpu_0_power_avg_w", None)
    energy_joules = power_avg_w * total_time if power_avg_w else None
    util_avg = stats.get("gpu_0_util_avg_pct", None)

    # Calcular throughput
    total_samples = num_iters * batch_size
    throughput = total_samples / total_time if total_time > 0 else 0


    # calcular eficiencia (Samples/Joule)
    if energy_joules and energy_joules > 0:
        eficiencia = total_samples / energy_joules  # Samples/J
    else:
        eficiencia = None
        

    print("\n=== Resultados Inferencia ResNet-50 NVIDIA===")
    print(f"Batch Size: {batch_size}")
    print(f"Pasadas totales (Iterations): {num_iters}")
    print(f"Total muestras: {total_samples}")
    print(f"Tiempo total: {total_time:.3f} s")
    print(f"Throughput: {throughput:.2f} Images/s")
    print(f"Potencia promedio: {power_avg_w:.3f} W")
    print(f"Energía total: {energy_joules:.3f} J")
    print(f"Eficiencia: {eficiencia:.3f} Images/J" if eficiencia else "No se pudo calcular eficiencia")
    print(f"Uso promedio: {util_avg:.3f}%")

    return {
        "num_iters": num_iters,
        "batch_size": batch_size,
        "total_time_s": total_time,
        "throughput_samples_s": throughput,
        "power_avg_w": power_avg_w,
        "energy_j": energy_joules,
        "efficiency_samples_per_joule": eficiencia
    }

monitor = GpuMonitor(interval=0.05, min_w_usage=10)

resultados = medir_eficiencia(model, monitor, num_iters=10000, batch_size=32)