#! /usr/bin/python3
# create_better_data.py
import pandas as pd

def create_better_financial_data():
    # More comprehensive and balanced financial sentiment data
    data = [
        # Positive examples
        {"text": "The stock market is showing strong bullish signals today", "label": 1},
        {"text": "Company earnings exceeded analyst expectations by 20%", "label": 1},
        {"text": "The Federal Reserve's decision boosted market confidence", "label": 1},
        {"text": "Dividend yields remain attractive for long-term investors", "label": 1},
        {"text": "The IPO was oversubscribed by institutional investors", "label": 1},
        {"text": "Strong cash flow generation supports future investments", "label": 1},
        {"text": "Market share expansion in key segments continues", "label": 1},
        {"text": "Innovation pipeline shows promising new products", "label": 1},
        {"text": "Record quarterly profits announced today", "label": 1},
        {"text": "Revenue growth accelerated this quarter", "label": 1},
        
        # Negative examples  
        {"text": "Market volatility is causing concern among investors", "label": 0},
        {"text": "Revenue growth slowed significantly this quarter", "label": 0},
        {"text": "Liquidity issues are affecting the company's operations", "label": 0},
        {"text": "Credit rating downgrade impacted bond prices negatively", "label": 0},
        {"text": "Supply chain disruptions are affecting profitability", "label": 0},
        {"text": "Debt levels are becoming unsustainable", "label": 0},
        {"text": "Regulatory changes pose significant risks", "label": 0},
        {"text": "Company announced major layoffs and restructuring", "label": 0},
        {"text": "Stock price plummeted after earnings miss", "label": 0},
        {"text": "Bankruptcy filing expected next week", "label": 0},
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('financial_sentiment_data.csv', index=False)
    print(f"Created {len(data)} balanced financial sentiment samples")
    print(f"Positive samples: {len(df[df['label'] == 1])}")
    print(f"Negative samples: {len(df[df['label'] == 0])}")
    
    return df

if __name__ == "__main__":
    create_better_financial_data()