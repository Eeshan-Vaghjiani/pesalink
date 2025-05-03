import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import subprocess
from PIL import Image

st.set_page_config(
    page_title="PesaLink Account Validation Dashboard",
    page_icon="💳",
    layout="wide"
)

# Constants
MODEL_METRICS_FILE = 'model_metrics.json'
CONFUSION_MATRIX_PATH = 'static/confusion_matrix.png'

# Helper functions
def load_latest_results():
    # Find the most recent results file
    result_files = glob.glob("validation_results_*.json")
    if not result_files:
        return None, None
    
    # Sort by modified time (newest first)
    latest_file = max(result_files, key=os.path.getmtime)
    
    # Find the corresponding summary file
    summary_file = latest_file.replace("results", "summary")
    
    results = None
    summary = None
    
    # Load the results
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            results = json.load(f)
    
    # Load the summary
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    return results, summary

def convert_results_to_dataframe(results):
    if not results:
        return pd.DataFrame()
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Extract any nested JSON data if present
    if 'data' in df.columns and df['data'].apply(lambda x: bool(x)).any():
        # Try to extract fields from the data column
        try:
            data_df = pd.json_normalize(df['data'])
            for col in data_df.columns:
                df[f'data_{col}'] = data_df[col]
        except:
            pass
            
    # Extract API status if available in the nested data
    if 'data' in df.columns and df['data'].apply(lambda x: isinstance(x, dict) and 'status' in x).any():
        df['apiStatus'] = df['data'].apply(lambda x: x.get('status', 'Unknown') if isinstance(x, dict) else 'Unknown')
    
    # Make sure required columns exist
    if 'apiStatus' not in df.columns and 'data_status' in df.columns:
        df['apiStatus'] = df['data_status']
        
    # Use the provided apiStatus field if it exists
    if 'apiStatus' not in df.columns:
        df['apiStatus'] = df.get('apiStatus', 'Unknown')
        
    # Ensure HTTP status field is available 
    if 'httpStatus' in df.columns:
        df['statusCode'] = df['httpStatus']
    
    # Ensure success rate is calculated correctly (based on both HTTP status and API status)
    if 'success' in df.columns:
        actual_success_rate = (df['success'] == True).mean() * 100
    else:
        # If no success field, calculate from HTTP and API status
        if 'statusCode' in df.columns and 'apiStatus' in df.columns:
            df['success'] = (df['statusCode'] == 200) & (df['apiStatus'] == 'Valid')
            actual_success_rate = df['success'].mean() * 100
        else:
            actual_success_rate = 0
    
    # Count success/failure based on the success field
    successful_count = df[df['success'] == True].shape[0]
    failed_count = df.shape[0] - successful_count
    
    return df

def load_model_metrics():
    if os.path.exists(MODEL_METRICS_FILE):
        with open(MODEL_METRICS_FILE, 'r') as f:
            return json.load(f)
    return None

def train_model():
    try:
        # Run the model training script
        result = subprocess.run(['python', 'ml_model.py'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        st.success("Model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error training model: {e.stderr}")
        return False

# Title and description
st.title("PesaLink Account Validation Dashboard")
st.markdown("""
This dashboard displays the results of bank account validations against the PesaLink API.
""")

# Load the data
try:
    results, summary = load_latest_results()

    if not results:
        st.warning("No validation results found. Please run the validation script first.")
        st.info("Run `node full_validation.js` to generate validation results")
    else:
        # Create DataFrames
        df = convert_results_to_dataframe(results)
        
        # Ensure success rate is calculated correctly
        if 'success' in df.columns:
            actual_success_rate = (df['success'] == True).mean() * 100
        else:
            actual_success_rate = 0
            
        # Calculate exact success rate (HTTP 200 only)
        exact_success_rate = (df['statusCode'] == 200).mean() * 100 if 'statusCode' in df.columns else actual_success_rate
        
        # Count occurrences by status code
        status_code_counts = {}
        if 'statusCode' in df.columns:
            status_code_counts = df['statusCode'].value_counts().to_dict()
        
        # Count success/failure
        successful_count = df[df['success'] == True].shape[0]
        failed_count = df.shape[0] - successful_count
        
        # Display summary metrics in a row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accounts", summary.get("total", 0))
        
        with col2:
            # Use the actual success rate from the data
            st.metric("Valid Accounts", f"{actual_success_rate:.2f}%")
        
        with col3:
            # Add structurally valid rate if available
            if 'isValidStructure' in df.columns:
                struct_valid_rate = (df['isValidStructure'] == True).mean() * 100
                st.metric("Structurally Valid", f"{struct_valid_rate:.2f}%")
            elif 'validStructureRate' in summary:
                # Try to get from summary
                struct_valid_rate = float(summary['validStructureRate'].replace('%', ''))
                st.metric("Structurally Valid", f"{struct_valid_rate:.2f}%")
            else:
                # Fall back to HTTP 200 status
                struct_valid_rate = (df['statusCode'] == 200).mean() * 100 if 'statusCode' in df.columns else 0
                st.metric("HTTP 200 Rate", f"{struct_valid_rate:.2f}%")
        
        with col4:
            st.metric("Invalid", failed_count)
        
        # Display timestamp of the data
        latest_file = max(glob.glob("validation_results_*.json"), key=os.path.getmtime)
        timestamp_str = latest_file.replace("validation_results_", "").replace(".json", "")
        st.caption(f"Last validation run: {timestamp_str}")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Validation Results", "Bank Analysis", "AI Model Evaluation"])
        
        with tab1:
            st.header("Validation Overview")
            
            # Success/failure pie chart
            st.subheader("Account Status Distribution")
            
            if successful_count > 0 or failed_count > 0:
                # Create tabs for different validity views
                validity_tabs = st.tabs(["API Status Distribution", "Functional Validity", "Structural Validity"])
                
                with validity_tabs[0]:
                    # Create detailed API status distribution chart
                    if 'apiStatus' in df.columns:
                        # Calculate count of each API status
                        api_status_counts = df['apiStatus'].value_counts().reset_index()
                        api_status_counts.columns = ['Status', 'Count']
                        api_status_counts['Percentage'] = (api_status_counts['Count'] / api_status_counts['Count'].sum() * 100).round(2)
                        
                        # Define colors for different statuses
                        status_colors = {
                            'Valid': '#4CAF50',      # Green
                            'Dormant': '#FF9800',    # Orange
                            'Post no Credit': '#FFC107', # Amber
                            'Invalid': '#F44336',    # Red
                            'Error': '#9C27B0',      # Purple
                            'Unknown': '#9E9E9E'     # Gray
                        }
                        
                        # Create API status distribution pie chart
                        fig = px.pie(
                            api_status_counts,
                            values='Count',
                            names='Status',
                            color='Status',
                            color_discrete_map=status_colors,
                            hole=0.4,
                            title="Account Status Distribution"
                        )
                        
                        fig.update_traces(
                            textinfo='percent+label',
                            textfont_size=14,
                            hoverinfo='label+percent+value',
                            marker=dict(line=dict(color='#000000', width=1))
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table with counts and percentages
                        st.dataframe(api_status_counts, use_container_width=True)
                    else:
                        st.info("No API status information available in the validation results")
                
                with validity_tabs[1]:
                    # Create pie chart for functional validity (Valid/Invalid)
                    # Recalculate using the exact success field values to ensure accuracy
                    valid_count = int(df['success'].sum())
                    invalid_count = int(df.shape[0] - valid_count)
                    
                    labels = ['Valid', 'Invalid']
                    values = [valid_count, invalid_count]
                    colors = ['#4CAF50', '#F44336']
                    
                    # Ensure values match the actual data
                    fig = px.pie(
                        values=values,
                        names=labels,
                        color_discrete_sequence=colors,
                        hole=0.4,
                        title=f"Functional Validity (status='Valid') - {actual_success_rate:.2f}% Valid"
                    )
                    # Add text annotation in the center with the success rate
                    fig.update_traces(
                        textinfo='percent',
                        textfont_size=14,
                        hoverinfo='label+percent',
                        marker=dict(line=dict(color='#000000', width=1))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with validity_tabs[2]:
                    # Create pie chart for structural validity (HTTP 200)
                    if 'isValidStructure' in df.columns:
                        struct_valid_count = df['isValidStructure'].sum()
                        struct_invalid_count = len(df) - struct_valid_count
                        
                        labels = ['Structurally Valid', 'Structurally Invalid']
                        values = [struct_valid_count, struct_invalid_count]
                        colors = ['#4CAF50', '#F44336']
                        
                        fig = px.pie(
                            values=values,
                            names=labels,
                            color_discrete_sequence=colors,
                            hole=0.4,
                            title=f"Structural Validity (HTTP 200) - {struct_valid_rate:.2f}% Valid"
                        )
                        fig.update_traces(
                            textinfo='percent',
                            textfont_size=14,
                            hoverinfo='label+percent',
                            marker=dict(line=dict(color='#000000', width=1))
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to HTTP status code
                        http_200_count = df[df['statusCode'] == 200].shape[0] if 'statusCode' in df.columns else 0
                        http_non_200_count = df.shape[0] - http_200_count
                        
                        labels = ['HTTP 200', 'Other HTTP Status']
                        values = [http_200_count, http_non_200_count]
                        colors = ['#4CAF50', '#F44336']
                        
                        fig = px.pie(
                            values=values,
                            names=labels,
                            color_discrete_sequence=colors,
                            hole=0.4,
                            title=f"HTTP Status Results - {struct_valid_rate:.2f}% HTTP 200"
                        )
                        fig.update_traces(
                            textinfo='percent',
                            textfont_size=14,
                            hoverinfo='label+percent',
                            marker=dict(line=dict(color='#000000', width=1))
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Create two columns for HTTP and API status
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                # HTTP status code distribution
                st.subheader("HTTP Status Codes")
                
                if 'statusCode' in df.columns:
                    status_counts = df['statusCode'].value_counts().reset_index()
                    status_counts.columns = ['Status Code', 'Count']
                    
                    # Add meaningful labels for common status codes
                    status_labels = {
                        200: 'OK - Success',
                        400: 'Bad Request',
                        404: 'Not Found',
                        500: 'Server Error'
                    }
                    
                    # Add labels to the dataframe
                    status_counts['Description'] = status_counts['Status Code'].map(
                        lambda x: status_labels.get(x, f'Status {x}')
                    )
                    
                    # Add percentage
                    status_counts['Percentage'] = status_counts['Count'] / status_counts['Count'].sum() * 100
                    
                    # Use Plotly for bar chart
                    fig = px.bar(
                        status_counts, 
                        x='Status Code', 
                        y='Count',
                        text='Description',
                        color='Count',
                        color_continuous_scale='RdYlGn_r',  # Red for high count, green for low count
                        title="HTTP Response Status Codes"
                    )
                    
                    fig.update_traces(
                        textposition='auto',
                        hovertemplate='Status %{x}: %{y} accounts<br>%{text}<br>'
                    )
                    
                    fig.update_layout(
                        xaxis_title="HTTP Status Code",
                        yaxis_title="Number of Accounts"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with status_col2:
                # API status distribution
                st.subheader("API Status Values")
                
                if 'apiStatus' in df.columns:
                    api_status_counts = df['apiStatus'].value_counts().reset_index()
                    api_status_counts.columns = ['API Status', 'Count']
                    
                    # Add percentage
                    api_status_counts['Percentage'] = api_status_counts['Count'] / api_status_counts['Count'].sum() * 100
                    
                    # Use Plotly for bar chart
                    fig = px.bar(
                        api_status_counts, 
                        x='API Status', 
                        y='Count',
                        color='Count',
                        color_continuous_scale='Viridis',
                        title="API Response Status Values"
                    )
                    
                    fig.update_layout(
                        xaxis_title="API Status",
                        yaxis_title="Number of Accounts"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Status Table
            st.subheader("Status Code Details")
            
            # Combine HTTP and API status with detailed counts
            if 'statusCode' in df.columns and 'apiStatus' in df.columns:
                st.markdown("#### Status Code Analysis")
                
                # Create a crosstab of HTTP Status vs API Status
                status_crosstab = pd.crosstab(df['statusCode'], df['apiStatus'])
                
                # Format for display
                st.dataframe(status_crosstab, use_container_width=True)
                
                # Also display a summary of invalid accounts
                if failed_count > 0:
                    st.markdown("#### Invalid Accounts Summary")
                    invalid_df = df[df['success'] == False].copy()
                    
                    # Group by reason if available
                    if 'errorMessage' in invalid_df.columns:
                        error_counts = invalid_df['errorMessage'].value_counts().reset_index()
                        error_counts.columns = ['Error Reason', 'Count']
                        error_counts['Percentage'] = error_counts['Count'] / error_counts['Count'].sum() * 100
                        
                        st.dataframe(error_counts, use_container_width=True)
            
            # Display logs if available
            st.subheader("Request Logs")
            logs_expander = st.expander("View Recent API Logs")
            with logs_expander:
                log_file = "./logs/requests_responses.log"
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        # Read last 10 lines of log file
                        lines = f.readlines()[-10:]
                        for line in lines:
                            try:
                                log_entry = json.loads(line.strip())
                                st.json(log_entry)
                            except:
                                st.text(line)
                else:
                    st.info("No logs available yet")
            
            # Display detailed information on API statuses if available
            if 'apiStatus' in df.columns:
                st.subheader("Account Status Analysis")
                
                # Create expander for detailed analysis
                status_expander = st.expander("Detailed Account Status Analysis", expanded=True)
                
                with status_expander:
                    # Get counts of each status
                    status_counts = df['apiStatus'].value_counts().reset_index()
                    status_counts.columns = ['Status', 'Count']
                    status_counts['Percentage'] = (status_counts['Count'] / status_counts['Count'].sum() * 100).round(2)
                    
                    # Create columns for visualization and table
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Display a table of status counts
                        st.dataframe(status_counts, use_container_width=True)
                        
                        # Create color map for known statuses
                        status_colors = {
                            'Valid': '#4CAF50',      # Green
                            'Invalid': '#F44336',    # Red
                            'Dormant': '#FF9800',    # Orange
                            'Post no Credit': '#FFC107', # Amber
                            'Unknown': '#9E9E9E'     # Gray
                        }
                        
                        # Assign colors based on known statuses, default to blue for unknown types
                        status_counts['Color'] = status_counts['Status'].apply(lambda x: status_colors.get(x, '#2196F3'))
                    
                    with col2:
                        # Create horizontal bar chart for status counts
                        fig = px.bar(
                            status_counts,
                            y='Status',
                            x='Count',
                            text='Percentage',
                            orientation='h',
                            color='Status',
                            color_discrete_map=status_colors,
                            title='Account Status Distribution'
                        )
                        
                        fig.update_traces(
                            texttemplate='%{text:.2f}%',
                            textposition='outside'
                        )
                        
                        fig.update_layout(
                            xaxis_title='Number of Accounts',
                            yaxis_title='Account Status',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate account status by bank (if we have enough data)
                    if df.shape[0] > 10 and 'bankCode' in df.columns:
                        st.subheader("Account Status by Bank")
                        
                        # Create a cross-tabulation of bank code and API status
                        bank_status_cross = pd.crosstab(df['bankCode'], df['apiStatus'])
                        
                        # Add a total column
                        bank_status_cross['Total'] = bank_status_cross.sum(axis=1)
                        
                        # Calculate percentages
                        bank_status_pct = bank_status_cross.iloc[:, :-1].div(bank_status_cross['Total'], axis=0) * 100
                        bank_status_pct = bank_status_pct.round(2)
                        
                        # Join the percentage dataframe with the total column
                        bank_status_pct['Total'] = bank_status_cross['Total']
                        
                        # Display the table
                        status_table_type = st.radio(
                            "Show counts or percentages:", 
                            ["Counts", "Percentages"], 
                            horizontal=True
                        )
                        
                        if status_table_type == "Counts":
                            st.dataframe(bank_status_cross, use_container_width=True)
                        else:
                            st.dataframe(bank_status_pct, use_container_width=True)
                        
                        # Create a heatmap of bank vs API status
                        bank_status_pct_reset = bank_status_pct.drop(columns=['Total']).reset_index()
                        bank_status_pct_melted = pd.melt(
                            bank_status_pct_reset,
                            id_vars=['bankCode'],
                            var_name='Status',
                            value_name='Percentage'
                        )
                        
                        # Create the heatmap
                        fig = px.density_heatmap(
                            bank_status_pct_melted,
                            x='bankCode',
                            y='Status',
                            z='Percentage',
                            color_continuous_scale='Viridis',
                            title='Account Status Distribution by Bank (%)'
                        )
                        
                        fig.update_layout(
                            xaxis_title='Bank Code',
                            yaxis_title='Account Status'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Validation Results")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.selectbox(
                    "Filter by Functional Validity",
                    ["All", "Valid", "Invalid"]
                )
            
            with col2:
                bank_codes = sorted(df['bankCode'].unique())
                bank_filter = st.multiselect(
                    "Filter by Bank Code",
                    bank_codes,
                    default=[]
                )
            
            with col3:
                search_term = st.text_input("Search Account Number")
            
            # API status filter if available
            if 'apiStatus' in df.columns:
                api_statuses = sorted(df['apiStatus'].unique())
                api_status_filter = st.multiselect(
                    "Filter by API Status",
                    api_statuses,
                    default=[]
                )
                
                # Add additional explanation about API status values
                st.caption("API Status Values: **Valid** (fully valid account), **Invalid** (non-existent account), **Dormant** (inactive account), **Post no Credit** (restricted account)")
            else:
                api_status_filter = []
            
            # Structural validity filter if available
            if 'isValidStructure' in df.columns:
                struct_validity_filter = st.radio(
                    "Filter by Structural Validity (HTTP 200)",
                    ["All", "Structurally Valid", "Structurally Invalid"],
                    horizontal=True
                )
            else:
                struct_validity_filter = "All"
            
            # Apply filters
            filtered_df = df.copy()
            
            if status_filter == "Valid":
                filtered_df = filtered_df[filtered_df['success'] == True]
            elif status_filter == "Invalid":
                filtered_df = filtered_df[filtered_df['success'] == False]
            
            if bank_filter:
                filtered_df = filtered_df[filtered_df['bankCode'].isin(bank_filter)]
            
            if api_status_filter:
                filtered_df = filtered_df[filtered_df['apiStatus'].isin(api_status_filter)]
            
            if search_term:
                filtered_df = filtered_df[filtered_df['accountNumber'].str.contains(search_term)]
            
            # Apply structural validity filter
            if struct_validity_filter == "Structurally Valid" and 'isValidStructure' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['isValidStructure'] == True]
            elif struct_validity_filter == "Structurally Invalid" and 'isValidStructure' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['isValidStructure'] == False]
            
            # Display dataframe with pagination
            display_columns = ['accountNumber', 'bankCode']
            
            # Add API status column if available
            if 'apiStatus' in filtered_df.columns:
                display_columns.append('apiStatus')
                
            # Add account holder name if available
            if 'accountHolderName' in filtered_df.columns:
                display_columns.append('accountHolderName')
                
            # Add bank name if available
            if 'bankName' in filtered_df.columns:
                display_columns.append('bankName')
                
            # Add currency if available
            if 'currency' in filtered_df.columns:
                display_columns.append('currency')
                
            # Add structural validity if available
            if 'isValidStructure' in filtered_df.columns:
                display_columns.append('isValidStructure')
                
            # Add HTTP status code
            if 'statusCode' in filtered_df.columns:
                display_columns.append('statusCode')
                
            # Add functional validity status    
            display_columns.append('success')
            
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True
            )
            
            # Display full JSON for selected account
            if not filtered_df.empty:
                selected_account = st.selectbox(
                    "Select an account to view details",
                    filtered_df['accountNumber'].tolist()
                )
                
                selected_data = filtered_df[filtered_df['accountNumber'] == selected_account].iloc[0].to_dict()
                st.json(selected_data)
        
        with tab3:
            st.header("Bank Analysis")
            
            # Create tabs for different validity metrics
            bank_validity_tabs = st.tabs(["Functional Validity", "Structural Validity", "Status Distribution"])
            
            with bank_validity_tabs[0]:
                # Group by bank code for functional validity
                bank_stats = df.groupby('bankCode').agg({
                    'success': ['count', lambda x: x.sum() / len(x) * 100],
                    'statusCode': lambda x: x.value_counts().to_dict()
                }).reset_index()
                
                bank_stats.columns = ['Bank Code', 'Total Accounts', 'Valid Rate (%)', 'Status Codes']
                
                # Format success rate
                bank_stats['Valid Rate (%)'] = bank_stats['Valid Rate (%)'].round(2)
                
                # Sort by success rate
                bank_stats = bank_stats.sort_values(by='Valid Rate (%)', ascending=False)
                
                # Display results
                st.dataframe(bank_stats[['Bank Code', 'Total Accounts', 'Valid Rate (%)']], use_container_width=True)
                
                # Success rate by bank chart using Plotly
                st.subheader("Valid Account Rate by Bank")
                fig = px.bar(
                    bank_stats, 
                    x='Bank Code', 
                    y='Valid Rate (%)',
                    color='Valid Rate (%)',
                    color_continuous_scale='RdYlGn',
                    title="Functional Validity Rate by Bank Code (status='Valid')",
                    labels={'Bank Code': 'Bank Code', 'Valid Rate (%)': 'Valid Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with bank_validity_tabs[1]:
                # Group by bank code for structural validity
                if 'isValidStructure' in df.columns:
                    struct_bank_stats = df.groupby('bankCode').agg({
                        'isValidStructure': ['count', lambda x: x.sum() / len(x) * 100]
                    }).reset_index()
                    
                    struct_bank_stats.columns = ['Bank Code', 'Total Accounts', 'Structurally Valid Rate (%)']
                    
                    # Format rate
                    struct_bank_stats['Structurally Valid Rate (%)'] = struct_bank_stats['Structurally Valid Rate (%)'].round(2)
                    
                    # Sort by rate
                    struct_bank_stats = struct_bank_stats.sort_values(by='Structurally Valid Rate (%)', ascending=False)
                    
                    # Display results
                    st.dataframe(struct_bank_stats, use_container_width=True)
                    
                    # Chart
                    st.subheader("Structurally Valid Account Rate by Bank")
                    fig = px.bar(
                        struct_bank_stats, 
                        x='Bank Code', 
                        y='Structurally Valid Rate (%)',
                        color='Structurally Valid Rate (%)',
                        color_continuous_scale='Blues',
                        title="Structural Validity Rate by Bank Code (HTTP 200)",
                        labels={'Bank Code': 'Bank Code', 'Structurally Valid Rate (%)': 'Structurally Valid Rate (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to HTTP 200 status
                    http_bank_stats = df.groupby('bankCode').agg({
                        'statusCode': ['count', lambda x: (x == 200).sum() / len(x) * 100]
                    }).reset_index()
                    
                    http_bank_stats.columns = ['Bank Code', 'Total Accounts', 'HTTP 200 Rate (%)']
                    
                    # Format rate
                    http_bank_stats['HTTP 200 Rate (%)'] = http_bank_stats['HTTP 200 Rate (%)'].round(2)
                    
                    # Sort by rate
                    http_bank_stats = http_bank_stats.sort_values(by='HTTP 200 Rate (%)', ascending=False)
                    
                    # Display results
                    st.dataframe(http_bank_stats, use_container_width=True)
                    
                    # Chart
                    st.subheader("HTTP 200 Rate by Bank")
                    fig = px.bar(
                        http_bank_stats, 
                        x='Bank Code', 
                        y='HTTP 200 Rate (%)',
                        color='HTTP 200 Rate (%)',
                        color_continuous_scale='Blues',
                        title="HTTP 200 Rate by Bank Code",
                        labels={'Bank Code': 'Bank Code', 'HTTP 200 Rate (%)': 'HTTP 200 Rate (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with bank_validity_tabs[2]:
                # Add API status by bank if available
                if 'apiStatus' in df.columns:
                    st.subheader("API Status by Bank")
                    
                    # Create a cross-tabulation of bank code vs API status
                    api_bank_cross = pd.crosstab(
                        df['bankCode'], 
                        df['apiStatus'], 
                        normalize='index'
                    ) * 100
                    
                    # Round to 2 decimal places
                    api_bank_cross = api_bank_cross.round(2)
                    
                    # Add a count column
                    bank_counts = df.groupby('bankCode').size().rename('Count')
                    api_bank_cross_with_count = pd.concat([api_bank_cross, bank_counts], axis=1)
                    
                    # Display the crosstab
                    st.dataframe(api_bank_cross_with_count, use_container_width=True)
                    
                    # Create a heatmap
                    api_bank_cross_reset = api_bank_cross.reset_index()
                    api_bank_cross_melted = pd.melt(
                        api_bank_cross_reset, 
                        id_vars=['bankCode'], 
                        var_name='API Status', 
                        value_name='Percentage'
                    )
                    
                    # Plot the heatmap
                    fig = px.density_heatmap(
                        api_bank_cross_melted,
                        x='bankCode',
                        y='API Status',
                        z='Percentage',
                        color_continuous_scale='YlGnBu',
                        title='API Status Distribution by Bank (%)'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Bank Code',
                        yaxis_title='API Status'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a bar chart showing counts of valid/invalid by bank
                    st.subheader("Valid vs Invalid Accounts by Bank")
                    
                    bank_validity = pd.crosstab(df['bankCode'], df['success'])
                    bank_validity.columns = ['Invalid', 'Valid']
                    bank_validity = bank_validity.reset_index()
                    
                    # Melt for easier plotting
                    bank_validity_melted = pd.melt(
                        bank_validity,
                        id_vars=['bankCode'],
                        var_name='Validity',
                        value_name='Count'
                    )
                    
                    # Create stacked bar chart
                    fig = px.bar(
                        bank_validity_melted,
                        x='bankCode',
                        y='Count',
                        color='Validity',
                        barmode='stack',
                        color_discrete_map={'Valid': '#4CAF50', 'Invalid': '#F44336'},
                        title='Valid and Invalid Accounts by Bank'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Bank Code',
                        yaxis_title='Number of Accounts'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add currency distribution analysis if currency field is available
                if 'currency' in df.columns:
                    st.subheader("Currency Distribution")
                    
                    # Create currency distribution chart
                    currency_counts = df['currency'].value_counts().reset_index()
                    currency_counts.columns = ['Currency', 'Count']
                    
                    # Only display if we have non-null currency values
                    if currency_counts['Count'].sum() > 0:
                        # Create pie chart for currency distribution
                        fig = px.pie(
                            currency_counts,
                            values='Count',
                            names='Currency',
                            title="Currency Distribution",
                            color_discrete_sequence=px.colors.qualitative.G10
                        )
                        
                        fig.update_traces(
                            textinfo='percent+label',
                            hoverinfo='label+percent+value'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show currency distribution by bank
                        st.subheader("Currency by Bank")
                        
                        # Create crosstab of bank vs currency
                        if 'bankCode' in df.columns:
                            bank_currency = pd.crosstab(df['bankCode'], df['currency'])
                            
                            # Calculate percentages by bank
                            bank_currency_pct = bank_currency.div(bank_currency.sum(axis=1), axis=0) * 100
                            bank_currency_pct = bank_currency_pct.round(2)
                            
                            # Display the table
                            st.dataframe(bank_currency, use_container_width=True)
                            
                            # Create a heatmap of bank vs currency
                            st.subheader("Currency Distribution by Bank (%)")
                            
                            # Reset index and melt for heatmap
                            bank_currency_pct_reset = bank_currency_pct.reset_index()
                            bank_currency_pct_melted = pd.melt(
                                bank_currency_pct_reset,
                                id_vars=['bankCode'],
                                var_name='Currency',
                                value_name='Percentage'
                            )
                            
                            # Create the heatmap
                            fig = px.density_heatmap(
                                bank_currency_pct_melted,
                                x='bankCode',
                                y='Currency',
                                z='Percentage',
                                color_continuous_scale='Viridis',
                                title='Currency Distribution by Bank (%)'
                            )
                            
                            fig.update_layout(
                                xaxis_title='Bank Code',
                                yaxis_title='Currency'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No currency data available in the validation results")
        
        with tab4:
            st.header("AI Model Evaluation")
            
            # Display ML model metrics
            model_metrics = load_model_metrics()
            
            if model_metrics:
                st.subheader("Model Performance Metrics")
                
                # Create metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.2f}")
                with col2:
                    st.metric("Precision", f"{model_metrics.get('precision', 0):.2f}")
                with col3:
                    st.metric("Recall", f"{model_metrics.get('recall', 0):.2f}")
                with col4:
                    st.metric("F1 Score", f"{model_metrics.get('f1_score', 0):.2f}")
                
                # Display confusion matrix
                if os.path.exists(CONFUSION_MATRIX_PATH):
                    st.subheader("Confusion Matrix")
                    confusion_matrix_img = Image.open(CONFUSION_MATRIX_PATH)
                    st.image(confusion_matrix_img, caption="Confusion Matrix")
                
                # Display ROC curve
                if os.path.exists('static/roc_curve.png'):
                    st.subheader("ROC Curve")
                    roc_curve_img = Image.open('static/roc_curve.png')
                    st.image(roc_curve_img, caption="ROC Curve")
                    
                # Display feature importance
                if os.path.exists('static/feature_importance.png'):
                    st.subheader("Feature Importance")
                    feature_importance_img = Image.open('static/feature_importance.png')
                    st.image(feature_importance_img, caption="Feature Importance")
            
            # Compare API validation results with ML predictions
            st.subheader("ML Predictions vs API Validation")
            
            # Check if there are ML predictions in the data
            if df is not None and 'ml_prediction' in df.columns:
                # Create a dataframe for the comparison
                ml_comparison_df = df[['accountNumber', 'bankCode', 'success', 'ml_prediction', 'ml_confidence', 'ml_prediction_correct']].copy()
                ml_comparison_df = ml_comparison_df.rename(columns={
                    'success': 'API Result',
                    'ml_prediction': 'ML Prediction',
                    'ml_confidence': 'ML Confidence (%)',
                    'ml_prediction_correct': 'Prediction Correct'
                })
                
                # Calculate ML prediction accuracy
                ml_accuracy = ml_comparison_df['Prediction Correct'].mean() * 100 if 'Prediction Correct' in ml_comparison_df.columns else 0
                
                st.metric("ML Prediction Accuracy", f"{ml_accuracy:.2f}%")
                
                # Filter options
                prediction_status = st.multiselect(
                    "Filter by prediction correctness:",
                    options=["Correct Predictions", "Incorrect Predictions"],
                    default=[]
                )
                
                filtered_df = ml_comparison_df
                
                if "Correct Predictions" in prediction_status and "Incorrect Predictions" not in prediction_status:
                    filtered_df = ml_comparison_df[ml_comparison_df['Prediction Correct'] == True]
                elif "Incorrect Predictions" in prediction_status and "Correct Predictions" not in prediction_status:
                    filtered_df = ml_comparison_df[ml_comparison_df['Prediction Correct'] == False]
                
                # Show the comparison table
                st.dataframe(filtered_df)
                
                # Display comparison chart
                st.subheader("ML Prediction Performance by Bank")
                
                # Group by bank code and calculate prediction accuracy
                if 'bankCode' in ml_comparison_df.columns and 'Prediction Correct' in ml_comparison_df.columns:
                    bank_performance = ml_comparison_df.groupby('bankCode')['Prediction Correct'].mean().reset_index()
                    bank_performance['Accuracy (%)'] = bank_performance['Prediction Correct'] * 100
                    bank_performance = bank_performance.sort_values('Accuracy (%)', ascending=False)
                    
                    fig = px.bar(
                        bank_performance,
                        x='bankCode',
                        y='Accuracy (%)',
                        title="ML Prediction Accuracy by Bank",
                        color='Accuracy (%)',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction distribution chart
                st.subheader("ML Prediction Distribution")
                
                prediction_counts = ml_comparison_df['ML Prediction'].value_counts().reset_index()
                prediction_counts.columns = ['Prediction', 'Count']
                
                fig = px.pie(
                    prediction_counts, 
                    values='Count', 
                    names='Prediction',
                    title="ML Prediction Distribution",
                    color_discrete_sequence=['#4CAF50', '#F44336']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                st.subheader("ML Prediction Confidence Distribution")
                
                # Create histogram of confidence values
                fig = px.histogram(
                    ml_comparison_df,
                    x='ML Confidence (%)',
                    color='Prediction Correct',
                    marginal="box",
                    nbins=20,
                    title="Distribution of ML Confidence Scores"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ML prediction data available in the current validation results. Run the fast ML prediction first with `python bulk_account_validator.py` then validate with `node full_validation.js --predictions <prediction_file>`")

            # Add a section for running predictions
            st.header("Run Fast Validation")

            cols = st.columns(2)
            with cols[0]:
                if st.button("Run ML Prediction", type="primary"):
                    with st.spinner("Running ML prediction..."):
                        result = subprocess.run(['python', 'bulk_account_validator.py'], 
                                              capture_output=True, 
                                              text=True)
                        
                        if result.returncode == 0:
                            st.success("ML prediction completed successfully!")
                            # Extract the prediction file from the output
                            output_lines = result.stdout.split('\n')
                            prediction_file = None
                            for line in output_lines:
                                if line.startswith("✅ ML predictions saved to"):
                                    prediction_file = line.replace("✅ ML predictions saved to ", "").strip()
                            
                            if prediction_file:
                                st.session_state.prediction_file = prediction_file
                                st.info(f"Predictions saved to {prediction_file}")
                                
                            st.code(result.stdout)
                        else:
                            st.error("Error running ML prediction")
                            st.code(result.stderr)

            with cols[1]:
                prediction_file = st.session_state.get('prediction_file', '')
                if prediction_file:
                    if st.button("Validate with API using ML Predictions", type="primary"):
                        with st.spinner("Running API validation with ML predictions..."):
                            result = subprocess.run(['node', 'full_validation.js', '--predictions', prediction_file], 
                                                  capture_output=True, 
                                                  text=True)
                            
                            if result.returncode == 0:
                                st.success("API validation completed successfully!")
                                st.code(result.stdout)
                                st.info("Refresh the page to see the updated results")
                            else:
                                st.error("Error running API validation")
                                st.code(result.stderr)
                else:
                    st.info("Run ML Prediction first to get prediction file")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("If you're seeing errors related to missing validation results, please run the validation script first:  \n```\nnode full_validation.js\n```")
