import pandas as pd
import numpy as np
import re
import joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURATION ===
INPUT_CSV = 'mock_bulk_transactions.csv'
USE_ML_MODEL = True  # Set to True to apply ML model
ML_MODEL_PATH = 'account_invalidity_predictor.joblib'

# === LOAD DATA ===
print('[INFO] Loading input CSV...')
df = pd.read_csv(INPUT_CSV)

# === RULE-BASED VALIDATION FUNCTION ===
def validate_account(row):
    account = str(row['account_number'])
    result = {'account_number': account,
              'bank_code': row['bank_code'],
              'amount': row['amount'],
              'reference_id': row['reference_id'],
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
    df_fe['log_amount'] = np.log1p(df_fe['amount'])
    df_fe['last_digit'] = df_fe['account_number'].astype(str).apply(lambda x: int(x[-1]))
    return df_fe

# === PARALLEL VALIDATION ===
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
