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
import multiprocessing
import time
from obtain_data import GpuMonitor

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

if os.path.exists(dataset_dir):
    for folder_name in os.listdir(dataset_dir):
        class_dir_0 = os.path.join(dataset_dir, folder_name, '0')
        class_dir_1 = os.path.join(dataset_dir, folder_name, '1')
        if os.path.exists(class_dir_0):
            for img_name in os.listdir(class_dir_0):
                image_paths.append(os.path.join(class_dir_0, img_name))
                labels.append(0)
        if os.path.exists(class_dir_1):
            for img_name in os.listdir(class_dir_1):
                image_paths.append(os.path.join(class_dir_1, img_name))
                labels.append(1)
    logger.info(f'Collected {len(image_paths)} images.')
else:
    logger.warning(f"Dataset directory not found: {dataset_dir}. Please ensure the path is correct.")

# Split the data into training, validation, and testing sets
logger.info('Splitting data into training, validation, and testing sets...')
if len(image_paths) > 0:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels, random_state=42)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.3333, stratify=temp_labels, random_state=42)

    logger.info(f'Training set size: {len(train_paths)}')
    logger.info(f'Validation set size: {len(val_paths)}')
    logger.info(f'Test set size: {len(test_paths)}')
else:
    logger.error("No images found. Exiting.")
    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

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
# Modifications: batch_size=256, num_workers>1 (setting to 4), pin_memory=True
BATCH_SIZE = 256
NUM_WORKERS = 4 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Early stopping
early_stopping_patience = 5
early_stopping_counter = 0
best_loss = float('inf')

# Training and evaluation
num_epochs = 10
best_model_wts = None

train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

# Initialize Monitor
monitor = GpuMonitor(interval=0.1, min_w_usage=40) # Monitor interval in seconds

logger.info('Starting training...')
monitor.clear()
if len(train_loader) > 0:
    total_images_processed = 0
    total_training_time = 0
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
                monitor.start()
                phase_start_time = time.time()
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                # Count images processed
                if phase == 'train':
                    total_images_processed += inputs.size(0)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                monitor.stop()
                phase_duration = time.time() - phase_start_time
                total_training_time += phase_duration

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu().numpy())
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc.cpu().numpy())
                val_loss_history.append(epoch_loss)

                # Deep copy the model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step(epoch_loss)

        if early_stopping_counter >= early_stopping_patience:
            logger.info('Early stopping triggered')
            break
else:
    logger.warning("Train loader is empty, skipping training loop.")

# monitor.stop() call is handled inside loop
total_time = total_training_time

logger.info('Training complete')

# --- Calculate and Print Statistics ---
stats = monitor.get_stats()
power_avg_w = stats.get("gpu_0_power_avg_w", None)
energy_joules = power_avg_w * total_time if power_avg_w else None
util_avg = stats.get("gpu_0_util_avg_pct_gfx", None)

# Calculate Efficiency
if energy_joules and energy_joules > 0:
    efficiency = total_images_processed / energy_joules  # Samples/J
else:
    efficiency = None

# Calculate Throughput
throughput = total_images_processed / total_time if total_time > 0 else 0

print("\n=== Training Results Resnet-50 AMD ===")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {num_epochs}")
print(f"Total Time: {total_time:.3f} s")
print(f"Throughput: {throughput:.2f} Images/s")
print(f"Average Power: {power_avg_w:.3f} W")
print(f"Total Energy: {energy_joules:.3f} J")
print(f"Efficiency: {efficiency:.3f} Images/J" if efficiency else "Efficiency: N/A")
print(f"Avg GPU Util: {util_avg:.3f} %")
print("========================")