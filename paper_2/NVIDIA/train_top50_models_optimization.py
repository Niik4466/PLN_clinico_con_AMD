#!/usr/bin/env python3
"""
Entrenamiento de Modelos de Clasificación ICD-10 - Top 50 Diagnósticos
VERSION OPTIMIZADA PARA NVIDIA (Tensor Cores, AMP, TF32, Flash Attention)

Con medición de eficiencia energética usando GPUMonitor

Entrena y guarda 4 arquitecturas de deep learning:
- SimpleRNN
- LSTM
- BiLSTM
- BERT (Bio_ClinicalBERT)

OPTIMIZACIONES HABILITADAS:
- Mixed Precision Training (AMP) con autocast + GradScaler
- TF32 para operaciones matmul en Tensor Cores
- cuDNN benchmark mode para autotuning
- SDPA (Scaled Dot-Product Attention) para Flash Attention en BERT
- torch.compile() opcional para PyTorch 2.0+

Autor: PHD Ilyas A.
Modificaciones por: Ignacio Ramírez
Versión Optimizada: Enero 2026
"""

import argparse
import warnings
import time
import csv
import os
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Importar GPUMonitor (instalado como paquete en Docker) ---
try:
    from GPUMonitor import GpuMonitor
    EFFICIENCY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: No se pudo importar GPUMonitor: {e}")
    EFFICIENCY_AVAILABLE = False

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

DATA_PATH = '/app/data/df_final_top50.csv'
MODELS_DIR = '/app/models'
BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
NUM_CLASSES = 50
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# ==============================================================================
# OPTIMIZACIONES NVIDIA
# ==============================================================================

def setup_nvidia_optimizations(use_tf32=True, use_cudnn_benchmark=True, use_compile=False):
    """
    Configura optimizaciones específicas para GPUs NVIDIA.
    
    Args:
        use_tf32: Habilitar TF32 para Tensor Cores (Ampere+)
        use_cudnn_benchmark: Habilitar cuDNN autotuning
        use_compile: Habilitar torch.compile() (PyTorch 2.0+)
    
    Returns:
        dict: Configuración de optimizaciones habilitadas
    """
    optimizations = {
        'tf32_matmul': False,
        'tf32_cudnn': False,
        'cudnn_benchmark': False,
        'compile_available': False,
        'amp_available': True,
        'sdpa_available': False,
    }
    
    # TF32 para Tensor Cores (NVIDIA Ampere y posterior)
    if use_tf32 and torch.cuda.is_available():
        # TF32 para operaciones matmul
        torch.backends.cuda.matmul.allow_tf32 = True
        optimizations['tf32_matmul'] = True
        # TF32 para cuDNN
        torch.backends.cudnn.allow_tf32 = True
        optimizations['tf32_cudnn'] = True
    
    # cuDNN benchmark mode - autotuning de algoritmos
    if use_cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        optimizations['cudnn_benchmark'] = True
    
    # Verificar si torch.compile está disponible (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        optimizations['compile_available'] = use_compile
    
    # Verificar SDPA (Scaled Dot-Product Attention) para Flash Attention
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        optimizations['sdpa_available'] = True
    
    return optimizations


def print_optimization_status(optimizations, device):
    """Imprime el estado de las optimizaciones."""
    print(f"\n{'='*80}")
    print(f"OPTIMIZACIONES NVIDIA HABILITADAS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"\n[✓] Mixed Precision (AMP): {optimizations['amp_available']}")
    print(f"[✓] TF32 MatMul: {optimizations['tf32_matmul']}")
    print(f"[✓] TF32 cuDNN: {optimizations['tf32_cudnn']}")
    print(f"[✓] cuDNN Benchmark: {optimizations['cudnn_benchmark']}")
    print(f"[✓] SDPA/Flash Attention: {optimizations['sdpa_available']}")
    print(f"[✓] torch.compile(): {optimizations['compile_available']}")
    print(f"{'='*80}\n")


# ==============================================================================
# DATASET
# ==============================================================================

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# ==============================================================================
# MODELOS
# ==============================================================================

class AdjustedRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.4):
        super(AdjustedRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        output = self.fc(self.dropout(rnn_out[:, -1, :]))
        return output


class AdjustedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super(AdjustedLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output


class AdjustedBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super(AdjustedBiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(self.dropout(
            torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
        ))
        return output


class BertClassifier(nn.Module):
    """
    BERT Classifier con soporte para SDPA (Flash Attention).
    """
    def __init__(self, bert_model_name, num_classes, use_sdpa=True):
        super(BertClassifier, self).__init__()
        
        # Cargar configuración y habilitar SDPA si está disponible
        config = AutoConfig.from_pretrained(bert_model_name)
        
        # Habilitar SDPA (Flash Attention) en transformers >= 4.36
        if use_sdpa and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            try:
                config.attn_implementation = "sdpa"
                print(f"[BERT] SDPA/Flash Attention habilitado")
            except Exception as e:
                print(f"[BERT] SDPA no disponible: {e}")
        
        self.bert = AutoModel.from_pretrained(bert_model_name, config=config)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# ==============================================================================
# FUNCIONES DE MEDICIÓN DE EFICIENCIA (con AMP)
# ==============================================================================


def medir_eficiencia_texto(model, model_name, monitor, device, vocab_size, num_classes, 
                           batch_size=16, seq_len=512, num_iters=100, is_bert=False,
                           use_amp=True, amp_dtype=torch.float16):
    """
    Mide eficiencia energética de modelos NLP con Mixed Precision.
    
    Args:
        use_amp: Usar Automatic Mixed Precision
        amp_dtype: Tipo de dato para AMP (float16 o bfloat16)
    """
    print(f"\n{'='*80}")
    print(f"MIDIENDO EFICIENCIA: {model_name}")
    print(f"{'='*80}")
    print(f"[AMP] Mixed Precision: {use_amp} ({amp_dtype})")
    
    model.eval()
    
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    if is_bert:
        dummy_attention_mask = torch.ones((batch_size, seq_len), device=device)
    
    # Warmup con AMP
    print("Warmup...")
    with torch.no_grad():
        with autocast(enabled=use_amp, dtype=amp_dtype):
            for _ in range(10):
                if is_bert:
                    _ = model(dummy_input_ids, dummy_attention_mask)
                else:
                    _ = model(dummy_input_ids)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
    
    monitor.clear()
    monitor.start()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        with autocast(enabled=use_amp, dtype=amp_dtype):
            for _ in tqdm(range(num_iters), desc=f"Benchmark {model_name}"):
                if is_bert:
                    _ = model(dummy_input_ids, dummy_attention_mask)
                else:
                    _ = model(dummy_input_ids)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    monitor.stop()
    
    stats = monitor.get_stats()
    power_avg_w = stats.get("gpu_0_power_avg_w", None)
    power_std_w = stats.get("gpu_0_power_std_w", 0)
    power_min_w = stats.get("gpu_0_power_min_w", None)
    power_max_w = stats.get("gpu_0_power_max_w", None)
    energy_joules = power_avg_w * total_time if power_avg_w else None
    
    # AMD usa gpu_0_util_avg_pct_gfx, NVIDIA usa gpu_0_util_avg_pct
    util_avg = stats.get("gpu_0_util_avg_pct", None)  # NVIDIA primero
    util_std = stats.get("gpu_0_util_std_pct", 0)
    util_min = stats.get("gpu_0_util_min_pct", None)
    util_max = stats.get("gpu_0_util_max_pct", None)
    if util_avg is None:
        util_avg = stats.get("gpu_0_util_avg_pct_gfx", None)  # AMD fallback
        util_std = stats.get("gpu_0_util_std_pct_gfx", 0)
        util_min = stats.get("gpu_0_util_min_pct_gfx", None)
        util_max = stats.get("gpu_0_util_max_pct_gfx", None)
    
    vram_avg_bytes = stats.get("gpu_0_vram_avg_bytes", None)
    mem_avg = vram_avg_bytes / (1024 * 1024) if vram_avg_bytes else None
    
    total_samples = num_iters * batch_size
    throughput = total_samples / total_time if total_time > 0 else 0
    
    # Eficiencia en samples por joule
    eficiencia = total_samples / energy_joules if energy_joules else None
    
    print(f"\n=== Resultados {model_name} ===")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Iteraciones: {num_iters}")
    print(f"Total muestras: {total_samples}")
    print(f"Tiempo total: {total_time:.3f} s")
    print(f"Throughput: {throughput:.2f} samples/s")
    if power_avg_w:
        print(f"Potencia promedio: {power_avg_w:.3f} W (±{power_std_w:.3f})")
    if energy_joules:
        print(f"Energía total: {energy_joules:.3f} J")
    if eficiencia:
        print(f"Eficiencia: {eficiencia:.2f} samples/J")
    if util_avg:
        print(f"Uso GPU promedio: {util_avg:.3f}% (±{util_std:.3f})")
    if mem_avg:
        print(f"Memoria GPU promedio: {mem_avg:.1f} MB")
    
    return {
        "model_name": model_name,
        "num_iters": num_iters,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_time_s": total_time,
        "throughput_samples_s": throughput,
        "power_avg_w": power_avg_w,
        "power_std_w": power_std_w,
        "power_min_w": power_min_w,
        "power_max_w": power_max_w,
        "energy_j": energy_joules,
        "efficiency_samples_per_joule": eficiencia,
        "gpu_util_avg_pct": util_avg,
        "gpu_util_std_pct": util_std,
        "gpu_util_min_pct": util_min,
        "gpu_util_max_pct": util_max,
        "gpu_mem_avg_mb": mem_avg,
        "amp_enabled": use_amp,
        "amp_dtype": str(amp_dtype),
    }


def guardar_resultados_eficiencia(resultados_list, output_path):
    """Guarda los resultados de eficiencia en un CSV."""
    if not resultados_list:
        return
    
    csv_path = os.path.join(output_path, "efficiency_results_optimized.csv")
    fieldnames = list(resultados_list[0].keys())
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultados_list)
    
    print(f"\nResultados de eficiencia guardados en: {csv_path}")


def export_model_info_texto(model, model_name, output_dir, batch_size, seq_len, optimizations=None):
    """Exporta información del modelo a un CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    output_path = os.path.join(output_dir, "model_info_optimized.csv")
    file_exists = os.path.isfile(output_path)
    
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model_name', 'input_shape', 'total_params', 'trainable_params', 
                           'dtype', 'amp_enabled', 'tf32_enabled', 'cudnn_benchmark'])
        writer.writerow([
            model_name,
            f"({batch_size}, {seq_len})",
            total_params,
            trainable_params,
            str(next(model.parameters()).dtype),
            optimizations.get('amp_available', False) if optimizations else False,
            optimizations.get('tf32_matmul', False) if optimizations else False,
            optimizations.get('cudnn_benchmark', False) if optimizations else False,
        ])
    
    print(f"Info del modelo {model_name} guardada en {output_path}")


# ==============================================================================
# FUNCIONES DE ENTRENAMIENTO CON AMP
# ==============================================================================

def train_rnn_lstm_bilstm(model, model_name, train_loader, test_loader, device, lr, save_path, 
                          num_epochs=10, early_stop_patience=3, use_amp=True, amp_dtype=torch.float16):
    """
    Entrena modelos RNN, LSTM o BiLSTM con early stopping, class weights y Mixed Precision.
    """
    print(f"\n{'='*80}")
    print(f"Entrenando {model_name} (AMP: {use_amp})")
    print(f"{'='*80}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    
    train_labels = [label for batch in train_loader for label in batch['labels'].tolist()]
    class_counts = Counter(train_labels)
    max_count = max(class_counts.values())
    class_weights = {class_id: max_count / count for class_id, count in class_counts.items()}
    weights = [class_weights[class_id] for class_id in sorted(class_weights.keys())]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        training_start_time = time.time()
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)  # Más eficiente que zero_grad()
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            
            # Forward pass con AMP
            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass con GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        training_time = time.time() - training_start_time
        
        model.eval()
        testing_start_time = time.time()
        predictions, true_labels = [], []
        val_loss = 0
        
        for batch in test_loader:
            with torch.no_grad():
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                    true_labels.extend(labels.cpu().tolist())
        
        testing_time = time.time() - testing_start_time
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        scheduler.step(val_loss / len(test_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Time: {training_time:.2f}s | Test Time: {testing_time:.2f}s | F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Modelo guardado (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping activado en epoch {epoch+1}")
                break
    
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=amp_dtype):
                inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                outputs = model(inputs)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
    
    report = classification_report(true_labels, predictions)
    print(f"\n{model_name} - Classification Report:")
    print(report)
    print(f"Best F1 Score: {best_f1:.4f}\n")
    
    return best_f1


def train_bert(model, train_loader, test_loader, device, save_path, 
               num_epochs=10, early_stop_patience=3, use_amp=True, amp_dtype=torch.float16):
    """
    Entrena modelo BERT con early stopping, class weights y Mixed Precision.
    """
    print(f"\n{'='*80}")
    print(f"Entrenando BERT (Bio_ClinicalBERT) - AMP: {use_amp}")
    print(f"{'='*80}")
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler(enabled=use_amp)
    
    train_labels = [label for batch in train_loader for label in batch['labels'].tolist()]
    class_counts = Counter(train_labels)
    max_count = max(class_counts.values())
    class_weights = {class_id: max_count / count for class_id, count in class_counts.items()}
    weights = [class_weights[class_id] for class_id in sorted(class_weights.keys())]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        training_start_time = time.time()
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass con AMP
            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            # Backward pass con GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        training_time = time.time() - training_start_time
        
        model.eval()
        testing_start_time = time.time()
        predictions, true_labels = [], []
        val_loss = 0
        
        for batch in test_loader:
            with torch.no_grad():
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask)
                    val_loss += criterion(outputs, labels).item()
                    predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                    true_labels.extend(labels.cpu().tolist())
        
        testing_time = time.time() - testing_start_time
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        scheduler.step(val_loss / len(test_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Time: {training_time:.2f}s | Test Time: {testing_time:.2f}s | F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Modelo guardado (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping activado en epoch {epoch+1}")
                break
    
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=amp_dtype):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
    
    report = classification_report(true_labels, predictions)
    print(f"\nBERT - Classification Report:")
    print(report)
    print(f"Best F1 Score: {best_f1:.4f}\n")
    
    return best_f1


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos Top50 ICD-10 (VERSION OPTIMIZADA NVIDIA)')
    parser.add_argument('--models', nargs='+', 
                        choices=['rnn', 'lstm', 'bilstm', 'bert', 'all'],
                        default=['all'],
                        help='Modelos a entrenar (default: all)')
    parser.add_argument('--data', type=str, default=DATA_PATH,
                        help=f'Ruta al dataset (default: {DATA_PATH})')
    parser.add_argument('--output', type=str, default=MODELS_DIR,
                        help=f'Directorio para guardar modelos (default: {MODELS_DIR})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Número de epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--efficiency-iters', type=int, default=100,
                        help='Número de iteraciones para benchmark de eficiencia (default: 100)')
    parser.add_argument('--skip-efficiency', action='store_true',
                        help='Saltar medición de eficiencia')
    
    # Argumentos de optimización
    parser.add_argument('--no-amp', action='store_true',
                        help='Deshabilitar Mixed Precision (AMP)')
    parser.add_argument('--no-tf32', action='store_true',
                        help='Deshabilitar TF32 para Tensor Cores')
    parser.add_argument('--no-cudnn-benchmark', action='store_true',
                        help='Deshabilitar cuDNN benchmark mode')
    parser.add_argument('--use-compile', action='store_true',
                        help='Habilitar torch.compile() (experimental)')
    parser.add_argument('--amp-dtype', type=str, choices=['float16', 'bfloat16'], default='float16',
                        help='Tipo de dato para AMP (default: float16)')
    
    args = parser.parse_args()
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    use_amp = not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurar optimizaciones NVIDIA
    optimizations = setup_nvidia_optimizations(
        use_tf32=not args.no_tf32,
        use_cudnn_benchmark=not args.no_cudnn_benchmark,
        use_compile=args.use_compile
    )
    optimizations['amp_available'] = use_amp
    
    print_optimization_status(optimizations, device)
    
    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data}")
    print(f"Models dir: {args.output}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Modelos a entrenar: {args.models}")
    print(f"Mixed Precision (AMP): {'Habilitado' if use_amp else 'Deshabilitado'}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"Medición de eficiencia: {'Deshabilitada' if args.skip_efficiency else 'Habilitada'}")
    if EFFICIENCY_AVAILABLE:
        print(f"GPUMonitor: Disponible")
    else:
        print(f"GPUMonitor: No disponible")
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    monitor = None
    if EFFICIENCY_AVAILABLE and not args.skip_efficiency:
        monitor = GpuMonitor(interval=0.05, min_w_usage=10)
    
    print(f"\n{'='*80}")
    print(f"CARGANDO DATASET")
    print(f"{'='*80}")
    df_final = pd.read_csv(args.data)
    print(f"Shape: {df_final.shape}")
    print(f"Columnas: {df_final.columns.tolist()}")
    
    assert 'symptoms' in df_final.columns, "Columna 'symptoms' no encontrada"
    assert 'labels' in df_final.columns, "Columna 'labels' no encontrada"
    
    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    print(f"\n{'='*80}")
    print(f"TOKENIZACIÓN")
    print(f"{'='*80}")
    print(f"Cargando tokenizer: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    def tokenize_texts(texts):
        return tokenizer(texts, padding='max_length', truncation=True, 
                        max_length=MAX_LENGTH, return_tensors='pt')
    
    print("Tokenizando train set...")
    train_encodings = tokenize_texts(train_df['symptoms'].tolist())
    print("Tokenizando test set...")
    test_encodings = tokenize_texts(test_df['symptoms'].tolist())
    
    train_dataset = TextDataset(train_encodings, train_df['labels'].tolist())
    test_dataset = TextDataset(test_encodings, test_df['labels'].tolist())
    
    # pin_memory para transferencias más rápidas GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['rnn', 'lstm', 'bilstm', 'bert']
    
    results = {}
    efficiency_results = []
    
    if 'rnn' in models_to_train:
        model = AdjustedRNNModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128, hidden_dim=256,
            output_dim=NUM_CLASSES, dropout=0.4
        ).to(device)
        
        # Opcional: torch.compile para PyTorch 2.0+
        if optimizations['compile_available'] and hasattr(torch, 'compile'):
            print("[torch.compile] Compilando SimpleRNN...")
            model = torch.compile(model)
        
        save_path = Path(args.output) / 'simplernn_model_optimized.pth'
        results['SimpleRNN'] = train_rnn_lstm_bilstm(
            model, 'SimpleRNN', train_loader, test_loader, device, 
            lr=0.00002, save_path=save_path, num_epochs=num_epochs,
            use_amp=use_amp, amp_dtype=amp_dtype
        )
        if monitor:
            eff = medir_eficiencia_texto(model, 'SimpleRNN', monitor, device, 
                                         vocab_size=tokenizer.vocab_size,
                                         num_classes=NUM_CLASSES,
                                         batch_size=batch_size, seq_len=MAX_LENGTH,
                                         num_iters=args.efficiency_iters, is_bert=False,
                                         use_amp=use_amp, amp_dtype=amp_dtype)
            efficiency_results.append(eff)
            export_model_info_texto(model, 'SimpleRNN', args.output, batch_size, MAX_LENGTH, optimizations)
    
    if 'lstm' in models_to_train:
        model = AdjustedLSTMModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128, hidden_dim=256,
            output_dim=NUM_CLASSES, dropout=0.2
        ).to(device)
        
        if optimizations['compile_available'] and hasattr(torch, 'compile'):
            print("[torch.compile] Compilando LSTM...")
            model = torch.compile(model)
        
        save_path = Path(args.output) / 'lstm_model_optimized.pth'
        results['LSTM'] = train_rnn_lstm_bilstm(
            model, 'LSTM', train_loader, test_loader, device,
            lr=0.001, save_path=save_path, num_epochs=num_epochs,
            use_amp=use_amp, amp_dtype=amp_dtype
        )
        if monitor:
            eff = medir_eficiencia_texto(model, 'LSTM', monitor, device,
                                         vocab_size=tokenizer.vocab_size,
                                         num_classes=NUM_CLASSES,
                                         batch_size=batch_size, seq_len=MAX_LENGTH,
                                         num_iters=args.efficiency_iters, is_bert=False,
                                         use_amp=use_amp, amp_dtype=amp_dtype)
            efficiency_results.append(eff)
            export_model_info_texto(model, 'LSTM', args.output, batch_size, MAX_LENGTH, optimizations)
    
    if 'bilstm' in models_to_train:
        model = AdjustedBiLSTMModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128, hidden_dim=256,
            output_dim=NUM_CLASSES, dropout=0.2
        ).to(device)
        
        if optimizations['compile_available'] and hasattr(torch, 'compile'):
            print("[torch.compile] Compilando BiLSTM...")
            model = torch.compile(model)
        
        save_path = Path(args.output) / 'bilstm_model_optimized.pth'
        results['BiLSTM'] = train_rnn_lstm_bilstm(
            model, 'BiLSTM', train_loader, test_loader, device,
            lr=0.001, save_path=save_path, num_epochs=num_epochs,
            use_amp=use_amp, amp_dtype=amp_dtype
        )
        if monitor:
            eff = medir_eficiencia_texto(model, 'BiLSTM', monitor, device,
                                         vocab_size=tokenizer.vocab_size,
                                         num_classes=NUM_CLASSES,
                                         batch_size=batch_size, seq_len=MAX_LENGTH,
                                         num_iters=args.efficiency_iters, is_bert=False,
                                         use_amp=use_amp, amp_dtype=amp_dtype)
            efficiency_results.append(eff)
            export_model_info_texto(model, 'BiLSTM', args.output, batch_size, MAX_LENGTH, optimizations)
    
    if 'bert' in models_to_train:
        # BERT con SDPA (Flash Attention)
        model = BertClassifier(BERT_MODEL_NAME, num_classes=NUM_CLASSES, 
                               use_sdpa=optimizations['sdpa_available']).to(device)
        
        if optimizations['compile_available'] and hasattr(torch, 'compile'):
            print("[torch.compile] Compilando BERT...")
            model = torch.compile(model)
        
        save_path = Path(args.output) / 'bert_model_optimized.pth'
        results['BERT'] = train_bert(model, train_loader, test_loader, device, save_path, 
                                     num_epochs=num_epochs, use_amp=use_amp, amp_dtype=amp_dtype)
        if monitor:
            eff = medir_eficiencia_texto(model, 'BERT', monitor, device,
                                         vocab_size=tokenizer.vocab_size,
                                         num_classes=NUM_CLASSES,
                                         batch_size=batch_size, seq_len=MAX_LENGTH,
                                         num_iters=args.efficiency_iters, is_bert=True,
                                         use_amp=use_amp, amp_dtype=amp_dtype)
            efficiency_results.append(eff)
            export_model_info_texto(model, 'BERT', args.output, batch_size, MAX_LENGTH, optimizations)
    
    if efficiency_results:
        guardar_resultados_eficiencia(efficiency_results, args.output)
    
    print(f"\n{'='*80}")
    print(f"RESUMEN DE RESULTADOS")
    print(f"{'='*80}")
    for model_name, f1 in results.items():
        print(f"{model_name:15s} | Best F1: {f1:.4f}")
    print(f"{'='*80}\n")
    print(f"Modelos guardados en: {args.output}")
    
    if efficiency_results:
        print(f"\n{'='*80}")
        print(f"RESUMEN DE EFICIENCIA (OPTIMIZADO)")
        print(f"{'='*80}")
        for eff in efficiency_results:
            throughput = eff.get('throughput_samples_s', 0)
            power = eff.get('power_avg_w', 'N/A')
            efficiency = eff.get('efficiency_samples_per_joule', 'N/A')
            if efficiency and efficiency != 'N/A':
                efficiency = f"{efficiency:.2f}"
            print(f"{eff['model_name']:15s} | Throughput: {throughput:,.1f} samples/s | Power: {power} W | Eff: {efficiency} samples/J")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
