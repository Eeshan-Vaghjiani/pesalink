import pandas as pd
import numpy as np
import re
import joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import json
import os
import time
from datetime import datetime

# === CONFIGURATION ===
INPUT_CSV = 'sample_1000_accounts.csv'  # Changed input file to the correct one
USE_ML_MODEL = False  # Set to False by default since we're focusing on prediction
ML_MODEL_PATH = 'account_invalidity_predictor.joblib'

# Add matplotlib configuration for visualization
plt.style.use('ggplot')
# Check environment variable to determine whether to show plots interactively
SHOW_PLOTS = os.environ.get('SHOW_PLOTS', 'TRUE').upper() != 'FALSE'
SAVE_PLOTS = True  # Always save plots to files

if not SHOW_PLOTS:
    # Use Agg backend when running headless (no GUI)
    import matplotlib
    matplotlib.use('Agg')

# === LOAD DATA FUNCTION ===
def load_and_process_data():
    print('[INFO] Loading input CSV...')
    if not os.path.exists(INPUT_CSV):
        print(f'[ERROR] Input file {INPUT_CSV} not found!')
        return None
    
    df = pd.read_csv(INPUT_CSV)
    return df

# === RULE-BASED VALIDATION FUNCTION ===
def validate_account(row):
    account = str(row['Account Number'])
    bank_code = row['Bank Code']
    result = {'account_number': account,
              'bank_code': bank_code,
              'status': '',
              'reason': ''}
    
    if not re.fullmatch(r'\d{10,12}', account):
        result['status'] = 'invalid'
        result['reason'] = 'Format Error'
        return result
    
    rnd = np.random.rand()
    if rnd <= 0.90:
        result['status'] = 'valid'
        result['reason'] = ''
    elif rnd <= 0.95:
        result['status'] = 'invalid'
        result['reason'] = 'Inactive Account'
    else:
        result['status'] = 'invalid'
        result['reason'] = 'Account Does Not Exist'
        
    return result

# === FEATURE ENGINEERING FUNCTION ===
def feature_engineering(df):
    df_fe = df.copy()
    df_fe['account_length'] = df_fe['account_number'].astype(str).apply(len)
    bank_freq = df_fe['bank_code'].value_counts()
    df_fe['bank_code_freq'] = df_fe['bank_code'].map(bank_freq)
    
    # Handle columns that may not exist in the DataFrame
    if 'amount' in df_fe.columns:
        df_fe['log_amount'] = np.log1p(df_fe['amount'])
    else:
        df_fe['log_amount'] = 0  # Provide a default value
        
    df_fe['last_digit'] = df_fe['account_number'].astype(str).apply(lambda x: int(x[-1]))
    return df_fe

# === LEGACY VALIDATION FUNCTION - KEEP FOR REFERENCE ===
def run_legacy_validation():
    df = load_and_process_data()
    if df is None:
        return
    
    print('[INFO] Running rule-based validation...')
    with ThreadPoolExecutor(max_workers=8) as executor:
        validated_results = list(executor.map(validate_account, [row for _, row in df.iterrows()]))

    results_df = pd.DataFrame(validated_results)

    # === OPTIONAL ML MODEL PREDICTION ===
    if USE_ML_MODEL:
        print('[INFO] Applying ML model for predictive validation...')
        features_df = feature_engineering(results_df)
        X = features_df[['account_length', 'bank_code_freq', 'log_amount', 'last_digit']]

        # Ensure the model path is valid
        try:
            model = joblib.load(ML_MODEL_PATH)
        except FileNotFoundError:
            print('[ERROR] Model not found. Please ensure account_invalidity_predictor.joblib exists.')
            exit(1)
        
        # Predict using the model
        ml_preds = model.predict(X)
        results_df['ml_prediction'] = ml_preds
        for i in results_df.index:
            if results_df.loc[i, 'status'] == 'valid' and results_df.loc[i, 'ml_prediction'] == 1:
                results_df.loc[i, 'status'] = 'invalid'
                results_df.loc[i, 'reason'] = 'ML-Flagged Anomaly'

        # === Confusion Matrix ===
        y_true = results_df['status'].apply(lambda x: 0 if x == 'valid' else 1)  # Convert status to numeric for y_true
        cm = confusion_matrix(y_true, ml_preds)

        # Accuracy
        accuracy = accuracy_score(y_true, ml_preds)
        print(f'[INFO] Model Accuracy: {accuracy * 100:.2f}%')

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Invalid', 'Valid'], yticklabels=['Invalid', 'Valid'], ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    # === SPLIT OUTPUT ===
    valid_accounts = results_df[results_df['status'] == 'valid'].drop(columns=['ml_prediction'] if USE_ML_MODEL else [])
    invalid_accounts = results_df[results_df['status'] == 'invalid'].drop(columns=['ml_prediction'] if USE_ML_MODEL else [])

    valid_accounts.to_csv('valid_accounts.csv', index=False)
    invalid_accounts.to_csv('invalid_accounts.csv', index=False)

    # === SUMMARY REPORT ===
    summary = {
        'Total Records': len(results_df),
        'Valid Accounts': len(valid_accounts),
        'Invalid Accounts': len(invalid_accounts),
    }

    invalid_reason_counts = invalid_accounts['reason'].value_counts().to_dict()
    summary.update({f'Invalid - {reason}': count for reason, count in invalid_reason_counts.items()})

    print('\n===== SUMMARY REPORT =====')
    for k, v in summary.items():
        print(f'{k}: {v}')

    print('\n[INFO] Outputs generated: valid_accounts.csv, invalid_accounts.csv')

# Load the ML model for prediction
MODEL_PATH = 'account_validation_model.joblib'

def load_prediction_model():
    """Load the trained ML model for prediction"""
    if os.path.exists(MODEL_PATH):
        print(f"Loading ML model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        print(f"Warning: ML model not found at {MODEL_PATH}")
        return None

def extract_features(account_number, bank_code):
    """Extract features for prediction from account number and bank code"""
    features = {}
    
    # Basic account features
    features['account_length'] = len(str(account_number))
    features['first_digit'] = int(str(account_number)[0]) if len(str(account_number)) > 0 else 0
    features['last_digit'] = int(str(account_number)[-1]) if len(str(account_number)) > 0 else 0
    
    # Bank code features
    features['bank_code'] = int(bank_code)
    
    # Account number patterns
    features['account_sum_digits'] = sum(int(d) for d in str(account_number) if d.isdigit())
    features['account_num_digits'] = sum(1 for d in str(account_number) if d.isdigit())
    
    # Pattern: Number of consecutive zeros
    features['consecutive_zeros'] = max([len(s) for s in str(account_number).split('0') if s == ''] or [0])
    
    # Pattern: Digit frequency variation
    digit_counts = {d: str(account_number).count(d) for d in '0123456789' if d in str(account_number)}
    features['digit_variance'] = np.var(list(digit_counts.values())) if digit_counts else 0
    
    # Position-based features
    for pos in range(min(4, features['account_length'])):
        features[f'digit_pos_{pos}'] = int(str(account_number)[pos]) if len(str(account_number)) > pos else -1
    
    return features

def predict_accounts_validity(accounts, model):
    """Predict account validity using ML model without API calls"""
    if not model:
        print("No ML model available for prediction")
        return accounts
    
    print(f"Predicting validity for {len(accounts)} accounts using ML model...")
    start_time = time.time()
    
    for account in accounts:
        # Extract features
        account_number = account.get('Account Number')
        bank_code = account.get('Bank Code')
        
        features_dict = extract_features(account_number, bank_code)
        
        # Convert to dataframe for prediction
        features_df = pd.DataFrame([features_dict])
        
        # Make prediction
        try:
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0][1]  # Probability of class 1 (valid)
            
            account['ml_prediction'] = bool(prediction)
            account['ml_confidence'] = round(max(prediction_proba, 1-prediction_proba) * 100, 2)
        except Exception as e:
            print(f"Error predicting for account {account_number}: {str(e)}")
            account['ml_prediction'] = None
            account['ml_confidence'] = 0
    
    end_time = time.time()
    print(f"Success: Prediction completed in {round(end_time - start_time, 2)} seconds")
    
    return accounts

def read_accounts_from_csv(file_path):
    """Read accounts from a CSV file"""
    accounts = []
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return accounts
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            accounts.append(dict(row))
    
    print(f"Read {len(accounts)} accounts from CSV file {file_path}")
    return accounts

def visualize_predictions(accounts):
    """Create visualizations for the prediction results"""
    if not accounts:
        print("No accounts to visualize")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(accounts)
    
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    try:
        # 1. Create prediction distribution pie chart
        valid_count = sum(1 for acc in accounts if acc.get('ml_prediction') == True)
        invalid_count = sum(1 for acc in accounts if acc.get('ml_prediction') == False)
        
        # Make sure we have data to plot
        if valid_count + invalid_count > 0:
            plt.figure(figsize=(10, 6))
            plt.pie([valid_count, invalid_count], 
                    labels=['Valid', 'Invalid'], 
                    autopct='%1.1f%%',
                    colors=['#4CAF50', '#F44336'],
                    shadow=False)  # Simplified - removed shadow and explode which can cause issues
            plt.title('ML Prediction Results Distribution')
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            
            if SAVE_PLOTS:
                plt.savefig('static/prediction_distribution.png', dpi=300, bbox_inches='tight')
                print(f"Saved prediction distribution chart to static/prediction_distribution.png")
            
            if SHOW_PLOTS:
                plt.show()
            plt.close()
        
        # 2. Create confidence histogram if ml_confidence exists
        if 'ml_confidence' in df.columns and not df['ml_confidence'].isna().all():
            plt.figure(figsize=(12, 6))
            df['ml_confidence'].hist(bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribution of ML Prediction Confidence')
            plt.xlabel('Confidence (%)')
            plt.ylabel('Number of Accounts')
            plt.grid(True, alpha=0.3)
            
            if SAVE_PLOTS:
                plt.savefig('static/confidence_histogram.png', dpi=300, bbox_inches='tight')
                print(f"Saved confidence histogram to static/confidence_histogram.png")
            
            if SHOW_PLOTS:
                plt.show()
            plt.close()
        
        # 3. Create prediction by bank code if Bank Code exists
        if 'Bank Code' in df.columns and 'ml_prediction' in df.columns:
            try:
                # Convert Bank Code to string for grouping
                df['Bank Code'] = df['Bank Code'].astype(str)
                
                # Group by bank code
                bank_valid_pct = df.groupby('Bank Code')['ml_prediction'].mean() * 100
                bank_counts = df.groupby('Bank Code').size()
                
                # Only create the plot if we have data
                if len(bank_valid_pct) > 0:
                    plt.figure(figsize=(14, 7))
                    bars = plt.bar(bank_valid_pct.index, bank_valid_pct.values, color='teal')
                    
                    # Add count labels above bars
                    for i, (bank, valid_pct) in enumerate(bank_valid_pct.items()):
                        count = bank_counts.get(bank, 0)
                        plt.text(i, valid_pct + 2, f'n={count}', ha='center')
                    
                    plt.title('Predicted Valid Account Percentage by Bank')
                    plt.xlabel('Bank Code')
                    plt.ylabel('Valid Accounts (%)')
                    plt.ylim(0, 105)  # Set y-axis limit with a little padding for labels
                    plt.grid(axis='y', alpha=0.3)
                    
                    if SAVE_PLOTS:
                        plt.savefig('static/predictions_by_bank.png', dpi=300, bbox_inches='tight')
                        print(f"Saved bank predictions chart to static/predictions_by_bank.png")
                    
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
            except Exception as e:
                print(f"Error creating bank code visualization: {str(e)}")
        
        print("Visualization complete")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        # Continue with the rest of the program even if visualization fails

def save_ml_predictions(accounts, output_file=None):
    """Save ML predictions to a JSON file"""
    if not output_file:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
        output_file = f"ml_predictions_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(accounts, f, indent=2)
    
    print(f"Success: ML predictions saved to {output_file}")
    return output_file

def main():
    # Load the prediction model
    model = load_prediction_model()
    
    # Read accounts from CSV
    accounts = read_accounts_from_csv('sample_1000_accounts.csv')
    if not accounts:
        return
    
    # Make fast predictions with ML model
    accounts_with_predictions = predict_accounts_validity(accounts, model)
    
    # Generate visualizations
    visualize_predictions(accounts_with_predictions)
    
    # Save predictions
    predictions_file = save_ml_predictions(accounts_with_predictions)
    
    # Display summary
    valid_count = sum(1 for acc in accounts_with_predictions if acc.get('ml_prediction') == True)
    invalid_count = sum(1 for acc in accounts_with_predictions if acc.get('ml_prediction') == False)
    total_count = len(accounts_with_predictions)
    
    print("\nPrediction Summary:")
    print(f"Total Accounts: {total_count}")
    print(f"Predicted Valid: {valid_count} ({round(valid_count/total_count*100, 2)}%)")
    print(f"Predicted Invalid: {invalid_count} ({round(invalid_count/total_count*100, 2)}%)")
    
    # Display visualization locations
    if SAVE_PLOTS:
        print("\nVisualizations saved to:")
        print("- static/prediction_distribution.png")
        print("- static/confidence_histogram.png")
        print("- static/predictions_by_bank.png")
    
    print(f"\nTo run API validation on these accounts, use: node full_validation.js --predictions {predictions_file}")

if __name__ == "__main__":
    main()
