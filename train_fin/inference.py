#! /usr/bin/python3
# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_fine_tuned_model():
    model_path = 'fine-tuned-financial-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        prediction = torch.argmax(outputs.logits, dim=1).cpu().item()
    
    # FIX: The model might have learned the opposite mapping
    # Let's check which class has higher confidence for positive-sounding text
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, probabilities, prediction

def main():
    # Load the fine-tuned model
    model, tokenizer, device = load_fine_tuned_model()
    print("Model loaded successfully!")
    
    # Test sentences with known expected outcomes
    test_sentences = [
        "The company reported excellent quarterly results",  # Should be Positive
        "Market conditions are deteriorating rapidly",       # Should be Negative
        "Revenue growth exceeded all expectations",          # Should be Positive  
        "The stock price dropped significantly today",       # Should be Negative
        "Strong fundamentals support future growth",         # Should be Positive
        "Bankruptcy filing announced yesterday",             # Should be Negative
        "Record profits this quarter",                       # Should be Positive
        "Massive layoffs expected next week",                # Should be Negative
    ]
    
    print("\nFinancial Sentiment Analysis Results:")
    print("=" * 60)
    
    # First, let's diagnose the issue
    print("Diagnosing model behavior...")
    positive_example = "Excellent earnings and growth"
    negative_example = "Bankruptcy and losses"
    
    pos_sentiment, pos_probs, pos_pred = predict_sentiment(positive_example, model, tokenizer, device)
    neg_sentiment, neg_probs, neg_pred = predict_sentiment(negative_example, model, tokenizer, device)
    
    print(f"\nPositive example: '{positive_example}'")
    print(f"Predicted: {pos_sentiment}, Class: {pos_pred}, Probs: {pos_probs}")
    
    print(f"\nNegative example: '{negative_example}'")
    print(f"Predicted: {neg_sentiment}, Class: {neg_pred}, Probs: {neg_probs}")
    
    # Check if we need to flip the predictions
    # if pos_pred == 0 and neg_pred == 1:  # Model learned opposite mapping
    #    print("\n⚠️  Model learned opposite label mapping! Flipping predictions...")
    #    flip_predictions = True
    # else:
    #    flip_predictions = False
    flip_predictions = False
    
    print("\n" + "=" * 60)
    print("Final Predictions:")
    print("=" * 60)
    
    for sentence in test_sentences:
        sentiment, confidence, pred_class = predict_sentiment(sentence, model, tokenizer, device)
        
        # Flip prediction if needed
        if flip_predictions:
            sentiment = "Negative" if sentiment == "Positive" else "Positive"
            pred_class = 1 - pred_class  # Flip 0->1, 1->0
        
        print(f"Text: {sentence}")
        print(f"Sentiment: {sentiment} (Class: {pred_class})")
        print(f"Confidence: [Negative: {confidence[0]:.4f}, Positive: {confidence[1]:.4f}]")
        print("-" * 40)

if __name__ == "__main__":
    main()