import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Bulk Account Validator", layout="wide")

st.title("💳 Bulk Account Validator — Pesalink Hackathon Demo")

st.sidebar.header("1️⃣ Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload bulk transaction CSV", type=['csv'])

USE_ML_MODEL = st.sidebar.checkbox('Use ML Model (optional)', value=True)

# === Helper functions ===
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

def feature_engineering(df):
    df_fe = df.copy()
    df_fe['account_length'] = df_fe['account_number'].astype(str).apply(len)
    bank_freq = df_fe['bank_code'].value_counts()
    df_fe['bank_code_freq'] = df_fe['bank_code'].map(bank_freq)
    df_fe['log_amount'] = np.log1p(df_fe['amount'])
    df_fe['last_digit'] = df_fe['account_number'].astype(str).apply(lambda x: int(x[-1]))
    return df_fe

def train_and_save_model(df, model_path):
    df_fe = feature_engineering(df)
    X = df_fe[['account_length', 'bank_code_freq', 'log_amount', 'last_digit']]
    y = df['status'].apply(lambda x: 0 if x == 'valid' else 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model, X_test, y_test

# === Main App Logic ===
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded {uploaded_file.name} with {len(df)} records")
    
    st.sidebar.header("2️⃣ Run Validation")
    if st.sidebar.button("Run Validation"):
        st.subheader("Running rule-based validation...")
        with st.spinner('Validating...'):
            with ThreadPoolExecutor(max_workers=8) as executor:
                validated_results = list(executor.map(validate_account, [row for _, row in df.iterrows()]))

            results_df = pd.DataFrame(validated_results)

            # Optional ML model
            if USE_ML_MODEL:
                st.subheader("Applying ML model for predictive validation...")
                model_path = 'account_invalidity_predictor.joblib'
                if not os.path.exists(model_path):
                    st.warning("ML model not found. Training a new one locally...")
                    # Simulate status labels from rule-based validation for training
                    results_df_for_model = results_df.copy()
                    model, X_test, y_test = train_and_save_model(results_df_for_model, model_path)
                else:
                    model = joblib.load(model_path)
                
                features_df = feature_engineering(results_df)
                X = features_df[['account_length', 'bank_code_freq', 'log_amount', 'last_digit']]
                ml_preds = model.predict(X)
                results_df['ml_prediction'] = ml_preds
                for i in results_df.index:
                    if results_df.loc[i, 'status'] == 'valid' and results_df.loc[i, 'ml_prediction'] == 1:
                        results_df.loc[i, 'status'] = 'invalid'
                        results_df.loc[i, 'reason'] = 'ML-Flagged Anomaly'

                # Confusion Matrix
                st.subheader("Confusion Matrix - Model Evaluation")
                cm = confusion_matrix(y_test, ml_preds)
                cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Valid', 'Invalid'])
                cm_disp.plot(cmap=plt.cm.Blues)
                st.pyplot()

            valid_accounts = results_df[results_df['status'] == 'valid']
            invalid_accounts = results_df[results_df['status'] == 'invalid']

            # === Summary Report ===
            st.header("📊 Validation Summary")
            total = len(results_df)
            valid = len(valid_accounts)
            invalid = len(invalid_accounts)
            st.metric(label="Total Records", value=total)
            col1, col2 = st.columns(2)
            col1.metric(label="✅ Valid Accounts", value=valid)
            col2.metric(label="❌ Invalid Accounts", value=invalid)

            # Pie chart
            fig1, ax1 = plt.subplots()
            ax1.pie([valid, invalid], labels=['Valid', 'Invalid'], autopct='%1.1f%%', startangle=90, colors=['#28a745', '#dc3545'])
            ax1.axis('equal')
            st.pyplot(fig1)

            # Bar chart of reasons
            st.subheader("Reasons for Invalid Accounts")
            reason_counts = invalid_accounts['reason'].value_counts()
            fig2, ax2 = plt.subplots()
            sns.barplot(x=reason_counts.index, y=reason_counts.values, palette='Set2', ax=ax2)
            plt.xticks(rotation=45)
            plt.ylabel('Count')
            st.pyplot(fig2)

            # Preview tables
            st.subheader("✅ Valid Accounts (Preview)")
            st.dataframe(valid_accounts.head())

            st.subheader("❌ Invalid Accounts (Preview)")
            st.dataframe(invalid_accounts.head())

            # Download buttons
            st.sidebar.header("3️⃣ Download Results")
            st.sidebar.download_button(label="Download Valid Accounts CSV", data=valid_accounts.to_csv(index=False), file_name='valid_accounts.csv', mime='text/csv')
            st.sidebar.download_button(label="Download Invalid Accounts CSV", data=invalid_accounts.to_csv(index=False), file_name='invalid_accounts.csv', mime='text/csv')

else:
    st.info("👆 Please upload a CSV file to begin")
