#! /usr/bin/python3
# train_better_model.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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

def train_better_model():
    # Load better data
    if not os.path.exists('financial_sentiment_data.csv'):
        print("Please run create_better_data.py first!")
        return
    
    df = pd.read_csv('financial_sentiment_data.csv')
    print(f"Data balance - Positive: {len(df[df['label'] == 1])}, Negative: {len(df[df['label'] == 0])}")
    
    # Split data with stratification
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42,
        stratify=df['label'].tolist()
    )
    
    # Load model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"},  # Explicit label mapping
        label2id={"Negative": 0, "Positive": 1}
    )
    model.to(device)
    
    # Data loaders
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Training
    epochs = 20  # Slightly more epochs
    learning_rate = 3e-5  # Slightly higher learning rate
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop with validation
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # Validate after each epoch
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                val_predictions.extend(predictions)
                val_true_labels.extend(labels)
        
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save_pretrained('fine-tuned-financial-model')
            tokenizer.save_pretrained('fine-tuned-financial-model')
            print(f"  -> New best model saved with accuracy: {best_accuracy:.4f}")
    
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    return model, tokenizer

if __name__ == "__main__":
    train_better_model()