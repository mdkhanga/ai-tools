#! /usr/bin/python3
# create_additional_data.py
import pandas as pd
import numpy as np

def create_additional_financial_data():
    # Sample financial sentiment data
    additional_data = [
        {"text": "The stock market is showing strong bullish signals today", "label": 1},
        {"text": "Company earnings exceeded analyst expectations", "label": 1},
        {"text": "Market volatility is causing concern among investors", "label": 0},
        {"text": "The Federal Reserve's decision boosted market confidence", "label": 1},
        {"text": "Revenue growth slowed significantly this quarter", "label": 0},
        {"text": "Dividend yields remain attractive for long-term investors", "label": 1},
        {"text": "Liquidity issues are affecting the company's operations", "label": 0},
        {"text": "The IPO was oversubscribed by institutional investors", "label": 1},
        {"text": "Credit rating downgrade impacted bond prices negatively", "label": 0},
        {"text": "Strong cash flow generation supports future investments", "label": 1},
        {"text": "Supply chain disruptions are affecting profitability", "label": 0},
        {"text": "Market share expansion in key segments continues", "label": 1},
        {"text": "Debt levels are becoming unsustainable", "label": 0},
        {"text": "Innovation pipeline shows promising new products", "label": 1},
        {"text": "Regulatory changes pose significant risks", "label": 0},
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(additional_data)
    
    # Save to CSV
    df.to_csv('financial_sentiment_data.csv', index=False)
    print(f"Created {len(additional_data)} additional financial sentiment samples")
    
    return df

if __name__ == "__main__":
    create_additional_financial_data()