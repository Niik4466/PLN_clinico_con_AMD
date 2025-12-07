#!/usr/bin/env python3
"""
Entrenamiento de Modelos de Clasificación ICD-10 - Top 10 Diagnósticos

Entrena y guarda 4 arquitecturas de deep learning:
- SimpleRNN
- LSTM
- BiLSTM
- BERT (Bio_ClinicalBERT)
Autor: PHD Ilyas A.
Modificaciones por: Ignacio Ramírez
Fecha: Octubre 2025
"""

import argparse
import warnings
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

DATA_PATH = '/app/data/df_final_top10.csv'
MODELS_DIR = '/app/models'
BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
NUM_CLASSES = 10
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

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
    def __init__(self, bert_model_name, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# ==============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ==============================================================================

def train_rnn_lstm_bilstm(model, model_name, train_loader, test_loader, device, lr, save_path, num_epochs=10, early_stop_patience=3):
    """
    Entrena modelos RNN, LSTM o BiLSTM con early stopping y class weights.
    """
    print(f"\n{'='*80}")
    print(f"Entrenando {model_name}")
    print(f"{'='*80}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Compute class weights
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
            optimizer.zero_grad()
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - training_start_time
        
        # Evaluation
        model.eval()
        testing_start_time = time.time()
        predictions, true_labels = [], []
        val_loss = 0
        
        for batch in test_loader:
            with torch.no_grad():
                inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        testing_time = time.time() - testing_start_time
        
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        scheduler.step(val_loss / len(test_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Time: {training_time:.2f}s | "
              f"Test Time: {testing_time:.2f}s | "
              f"F1: {val_f1:.4f}")
        
        # Early stopping
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
    
    # Final report
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        with torch.no_grad():
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    report = classification_report(true_labels, predictions)
    print(f"\n{model_name} - Classification Report:")
    print(report)
    print(f"Best F1 Score: {best_f1:.4f}\n")
    
    return best_f1


def train_bert(model, train_loader, test_loader, device, save_path, num_epochs=10, early_stop_patience=3):
    """
    Entrena modelo BERT con early stopping y class weights.
    """
    print(f"\n{'='*80}")
    print(f"Entrenando BERT (Bio_ClinicalBERT)")
    print(f"{'='*80}")
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # Compute class weights
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
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - training_start_time
        
        # Evaluation
        model.eval()
        testing_start_time = time.time()
        predictions, true_labels = [], []
        val_loss = 0
        
        for batch in test_loader:
            with torch.no_grad():
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
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Time: {training_time:.2f}s | "
              f"Test Time: {testing_time:.2f}s | "
              f"F1: {val_f1:.4f}")
        
        # Early stopping
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
    
    # Final report
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        with torch.no_grad():
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
    parser = argparse.ArgumentParser(description='Entrenar modelos Top10 ICD-10')
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
    
    args = parser.parse_args()
    
    # Usar los valores del parser
    num_epochs = args.epochs
    batch_size = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data}")
    print(f"Models dir: {args.output}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Modelos a entrenar: {args.models}")
    
    # Crear directorio de modelos si no existe
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Cargar dataset
    print(f"\n{'='*80}")
    print(f"CARGANDO DATASET")
    print(f"{'='*80}")
    df_final = pd.read_csv(args.data)
    print(f"Shape: {df_final.shape}")
    print(f"Columnas: {df_final.columns.tolist()}")
    
    assert 'symptoms' in df_final.columns, "Columna 'symptoms' no encontrada"
    assert 'labels' in df_final.columns, "Columna 'labels' no encontrada"
    
    # Train/test split (80/20)
    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Tokenización
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
    
    # Crear dataloaders
    train_dataset = TextDataset(train_encodings, train_df['labels'].tolist())
    test_dataset = TextDataset(test_encodings, test_df['labels'].tolist())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Determinar modelos a entrenar
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['rnn', 'lstm', 'bilstm', 'bert']
    
    results = {}
    
    # Entrenar modelos
    if 'rnn' in models_to_train:
        model = AdjustedRNNModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=NUM_CLASSES,
            dropout=0.4
        ).to(device)
        save_path = Path(args.output) / 'simplernn_model.pth'
        results['SimpleRNN'] = train_rnn_lstm_bilstm(
            model, 'SimpleRNN', train_loader, test_loader, device, 
            lr=0.00002, save_path=save_path, num_epochs=num_epochs
        )
    
    if 'lstm' in models_to_train:
        model = AdjustedLSTMModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=NUM_CLASSES,
            dropout=0.2
        ).to(device)
        save_path = Path(args.output) / 'lstm_model.pth'
        results['LSTM'] = train_rnn_lstm_bilstm(
            model, 'LSTM', train_loader, test_loader, device,
            lr=0.001, save_path=save_path, num_epochs=num_epochs
        )
    
    if 'bilstm' in models_to_train:
        model = AdjustedBiLSTMModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=NUM_CLASSES,
            dropout=0.2
        ).to(device)
        save_path = Path(args.output) / 'bilstm_model.pth'
        results['BiLSTM'] = train_rnn_lstm_bilstm(
            model, 'BiLSTM', train_loader, test_loader, device,
            lr=0.001, save_path=save_path, num_epochs=num_epochs
        )
    
    if 'bert' in models_to_train:
        model = BertClassifier(BERT_MODEL_NAME, num_classes=NUM_CLASSES).to(device)
        save_path = Path(args.output) / 'bert_model.pth'
        results['BERT'] = train_bert(model, train_loader, test_loader, device, save_path, num_epochs=num_epochs)
    
    # Resumen final
    print(f"\n{'='*80}")
    print(f"RESUMEN DE RESULTADOS")
    print(f"{'='*80}")
    for model_name, f1_score in results.items():
        print(f"{model_name:15s} | Best F1: {f1_score:.4f}")
    print(f"{'='*80}\n")

    print(f"Modelos guardados en: {args.output}")


if __name__ == '__main__':
    main()
