#! /usr/bin/python3
# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        prediction = torch.argmax(outputs.logits, dim=1).cpu().item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    return sentiment, confidence

def main():
    # Load the fine-tuned model
    model, tokenizer, device = load_fine_tuned_model()
    print("Model loaded successfully!")
    
    # Test sentences
    test_sentences = [
        "The company reported excellent quarterly results",
        "Market conditions are deteriorating rapidly",
        "Revenue growth exceeded all expectations",
        "The stock price dropped significantly today",
        "Strong fundamentals support future growth"
    ]
    
    print("\nFinancial Sentiment Analysis Results:")
    print("=" * 50)
    
    for sentence in test_sentences:
        sentiment, confidence = predict_sentiment(sentence, model, tokenizer, device)
        print(f"Text: {sentence}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence}")
        print("-" * 30)

if __name__ == "__main__":
    main()