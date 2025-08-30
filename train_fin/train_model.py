#! /usr/bin/python3
# train_model.py
# train_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW  # Correct import location for AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# Check PyTorch version and MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Set device - handle MPS availability properly
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Using device: {device}")

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data():
    # Load our created data
    if not os.path.exists('financial_sentiment_data.csv'):
        print("Creating sample data first...")
        # Create some sample data if file doesn't exist
        sample_data = {
            'text': [
                'Stock market is rising',
                'Company profits are down',
                'Great earnings report',
                'Market crash expected',
                'Strong financial results',
                'Revenue decline continues'
            ],
            'label': [1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv('financial_sentiment_data.csv', index=False)
    else:
        df = pd.read_csv('financial_sentiment_data.csv')
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42,
        stratify=df['label'].tolist()  # Maintain class balance
    )
    
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    return train_texts, val_texts, train_labels, val_labels

def train_model():
    # Load a small pre-trained model
    model_name = "distilbert-base-uncased"  # Small and efficient
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.to(device)
    print("Model loaded and moved to device")
    
    # Load data
    train_texts, val_texts, train_labels, val_labels = load_and_prepare_data()
    
    # Create datasets
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders with smaller batch size for Mac
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Smaller batch for Mac
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Training parameters - reduced for faster training on Mac
    epochs = 2  # Reduced from 3
    learning_rate = 2e-5
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Print progress every few batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    output_dir = 'fine-tuned-financial-model'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully to {output_dir}!")
    
    return model, tokenizer

def evaluate_model(model, tokenizer):
    # Load validation data
    df = pd.read_csv('financial_sentiment_data.csv')
    _, val_texts, _, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    model.eval()
    predictions = []
    true_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for i, (text, label) in enumerate(zip(val_texts, val_labels)):
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            pred = torch.argmax(outputs.logits, dim=1).cpu().item()
            predictions.append(pred)
            true_labels.append(label)
            
            if i < 3:  # Show first few predictions
                sentiment = "Positive" if pred == 1 else "Negative"
                actual = "Positive" if label == 1 else "Negative"
                print(f"Sample {i+1}: Predicted: {sentiment}, Actual: {actual}")
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    return accuracy

if __name__ == "__main__":
    try:
        # Train the model
        model, tokenizer = train_model()
        
        # Evaluate the model
        evaluate_model(model, tokenizer)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure you have updated packages: pip install --upgrade torch transformers")