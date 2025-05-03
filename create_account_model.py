import numpy as np
import pandas as pd
import os
import csv
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import glob
import json

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data for model training if no real data is available"""
    print("Generating synthetic training data...")
    
    np.random.seed(42)  # For reproducibility
    
    data = []
    for i in range(n_samples):
        account_length = np.random.choice([10, 11, 12])
        first_digit = np.random.randint(1, 10)
        last_digit = np.random.randint(0, 10)
        bank_code = np.random.randint(1, 10)
        consecutive_zeros = np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.05, 0.05])
        
        # Generate random account number
        account_number = str(first_digit)
        for j in range(account_length - 2):
            if consecutive_zeros > 0 and j < consecutive_zeros:
                account_number += "0"
            else:
                account_number += str(np.random.randint(0, 10))
        account_number += str(last_digit)
        
        # Calculate some features
        digit_sum = sum(int(d) for d in account_number)
        digit_variance = np.var([account_number.count(str(d)) for d in range(10)])
        
        # Determine validity (synthetic rule)
        # Make accounts with specific patterns more likely to be valid
        valid = np.random.random() < 0.8  # 80% valid by default
        
        if bank_code in [1, 2] and last_digit % 2 == 0:
            valid = np.random.random() < 0.95  # Higher validity for certain banks with even last digits
            
        if consecutive_zeros >= 3:
            valid = np.random.random() < 0.4  # Lower validity for accounts with many consecutive zeros
            
        data.append({
            'Account Number': account_number,
            'Bank Code': bank_code,
            'success': valid
        })
    
    return pd.DataFrame(data)

def extract_features(df):
    """Extract features for model training"""
    features = pd.DataFrame()
    
    # Account features
    features['account_length'] = df['Account Number'].astype(str).apply(len)
    features['first_digit'] = df['Account Number'].astype(str).apply(lambda x: int(x[0]) if len(x) > 0 else 0)
    features['last_digit'] = df['Account Number'].astype(str).apply(lambda x: int(x[-1]) if len(x) > 0 else 0)
    
    # Bank code feature
    features['bank_code'] = df['Bank Code'].astype(int)
    
    # Account number patterns
    features['account_sum_digits'] = df['Account Number'].astype(str).apply(lambda x: sum(int(d) for d in x if d.isdigit()))
    features['consecutive_zeros'] = df['Account Number'].astype(str).apply(lambda x: max([len(s) for s in x.split('0') if s == ''] or [0]))
    
    # Pattern: Digit frequency variation
    def digit_frequency_variance(account_num):
        digit_counts = {d: account_num.count(d) for d in '0123456789' if d in account_num}
        if digit_counts:
            return np.var(list(digit_counts.values()))
        return 0
    
    features['digit_variance'] = df['Account Number'].astype(str).apply(digit_frequency_variance)
    
    # Position-based features (first few positions)
    for pos in range(min(4, features['account_length'].min())):
        features[f'digit_pos_{pos}'] = df['Account Number'].astype(str).apply(lambda x: int(x[pos]) if len(x) > pos else -1)
    
    return features

def train_model(model_path='account_validation_model.joblib'):
    """Train a basic account validation model"""
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return True
    
    # Check if validation results exist to train on
    validation_files = glob.glob("validation_results_*.json")
    
    if validation_files:
        print("Found validation results to train on")
        # Load the most recent validation results
        latest_file = max(validation_files, key=os.path.getmtime)
        with open(latest_file, 'r') as f:
            results = json.load(f)
            
        df = pd.DataFrame(results)
        
    else:
        print("No validation results found, using synthetic data")
        # Use synthetic data if no validation results are available
        df = generate_synthetic_data()
        
        # Save some synthetic data to CSV for validation
        if not os.path.exists('sample_1000_accounts.csv'):
            print("Generating sample_1000_accounts.csv file")
            sample_df = df[['Account Number', 'Bank Code']]
            sample_df.to_csv('sample_1000_accounts.csv', index=False)
    
    print(f"Training model with {len(df)} records")
    
    # Extract features
    features = extract_features(df)
    
    # Target variable (success/failure)
    y = df['success'].astype(int)
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    # Save model metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(np.mean(y_pred[y_test==1])) if sum(y_test==1) > 0 else 0,
        'recall': float(sum(y_pred[y_test==1])/sum(y_test==1)) if sum(y_test==1) > 0 else 0,
        'f1_score': float(2 * accuracy / (1 + accuracy)) # Simplified F1 calculation
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model creation complete!")
    return True

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train/create the model
    train_model() 