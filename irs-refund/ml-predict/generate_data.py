#!/usr/bin/python3

import numpy as np
import pandas as pd

# Generate synthetic dataset
def generate_synthetic_data(n_samples=10000, random_state=42):
    np.random.seed(random_state)

    # Example features (like tax refund prediction inputs)
    income = np.random.normal(60000, 15000, n_samples)       # yearly income
    dependents = np.random.randint(0, 4, n_samples)          # number of dependents
    withholding = np.random.normal(5000, 2000, n_samples)    # taxes withheld
    filing_status = np.random.choice([0, 1], size=n_samples) # 0 = single, 1 = married

    # Target variable: "days until refund" (label)
    refund_days = (
        60
        - (withholding / 1000)
        + (np.random.normal(0, 5, n_samples))
        + dependents * 2
        - filing_status * 3
    )

    df = pd.DataFrame({
        "income": income,
        "dependents": dependents,
        "withholding": withholding,
        "filing_status": filing_status,
        "refund_days": refund_days
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("synthetic_refunds.csv", index=False)
    print("✅ Synthetic data saved to synthetic_refunds.csv")
