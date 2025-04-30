import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load already validated result (from your previous run)
df = pd.read_csv('mock_bulk_transactions.csv')

# Manually simulate the same labels
def simulate_validation(row):
    account = str(row['account_number'])
    if not account.isdigit() or not (10 <= len(account) <= 12):
        return 'invalid'
    rnd = np.random.rand()
    if rnd <= 0.90:
        return 'valid'
    elif rnd <= 0.95:
        return 'invalid'
    else:
        return 'invalid'

df['status'] = df.apply(simulate_validation, axis=1)

# Feature Engineering
df['account_length'] = df['account_number'].astype(str).apply(len)
bank_freq = df['bank_code'].value_counts()
df['bank_code_freq'] = df['bank_code'].map(bank_freq)
df['log_amount'] = np.log1p(df['amount'])
df['last_digit'] = df['account_number'].astype(str).apply(lambda x: int(x[-1]))

# Features + Labels
X = df[['account_length', 'bank_code_freq', 'log_amount', 'last_digit']]
y = df['status'].apply(lambda x: 0 if x == 'valid' else 1)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save locally (in same version)
joblib.dump(model, 'account_invalidity_predictor.joblib')

print('[INFO] Model trained and saved as account_invalidity_predictor.joblib')
