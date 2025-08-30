# main.py
import subprocess
import sys

def run_pipeline():
    print("Step 1: Creating additional financial data...")
    subprocess.run([sys.executable, "create_additional_data.py"])
    
    print("\nStep 2: Training the model...")
    subprocess.run([sys.executable, "train_model.py"])
    
    print("\nStep 3: Running inference on new data...")
    subprocess.run([sys.executable, "inference.py"])

if __name__ == "__main__":
    run_pipeline()