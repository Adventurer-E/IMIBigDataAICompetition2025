"""
main.py
-------
This script orchestrates the production pipeline:
1. Data Preparation
2. Feature Engineering
3. Model Training

It reads input data from /mnt/data (mounted as read-only) and writes output to /mnt/output.
"""

import subprocess

def run_pipeline():
    # Run data preparation
    subprocess.run(["python", "src/data_prep.py"], check=True)
    
    # Run feature engineering
    subprocess.run(["python", "src/feature_engineering.py"], check=True)
    
    # Run model training
    subprocess.run(["python", "src/model.py"], check=True)

if __name__ == "__main__":
    run_pipeline()
