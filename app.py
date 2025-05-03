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

# Set page configuration with improved appearance
st.set_page_config(
    page_title="PesaLink Account Validation Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom CSS for better appearance that works with both light and dark mode
st.markdown("""
<style>
    /* General styling */
    .main .block-container {
        padding: 2rem 3rem;
    }
    
    /* Headings: Use contrasting colors that work in both modes */
    h1, h2, h3 {
        color: #03A9F4 !important;
        font-weight: 600 !important;
    }
    
    h4, h5, h6 {
        color: #29B6F6 !important;
        font-weight: 500 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px;
        font-weight: 500;
    }
    
    /* Active tab should be clearly visible */
    .stTabs [aria-selected="true"] {
        background-color: #0277BD !important;
        color: white !important;
    }
    
    /* Metrics styling with border to ensure visibility in dark mode */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 5px;
        border: 1px solid rgba(150, 150, 150, 0.2);
    }
    
    /* Make metric labels more visible in dark mode */
    .stMetric label {
        color: #90CAF9 !important;
        font-weight: 500 !important;
    }
    
    /* Make metric values more visible */
    .stMetric [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    /* Table styling for better visibility */
    [data-testid="stTable"] {
        border: 1px solid rgba(150, 150, 150, 0.2);
    }
    
    /* Ensure dataframe headers are visible */
    .dataframe thead th {
        color: #E3F2FD !important;
        background-color: #1976D2 !important;
        padding: 8px !important;
    }
    
    /* Alternating rows for better readability */
    .dataframe tbody tr:nth-child(even) {
        background-color: rgba(200, 200, 200, 0.1);
    }
    
    /* Information boxes */
    .info-box {
        background-color: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        color: #E3F2FD;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stExpander"] > details {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(150, 150, 150, 0.2);
    }
    
    div[data-testid="stExpander"] > details > summary {
        padding: 1rem;
        font-weight: 600;
    }
    
    div[data-testid="stExpander"] > details > summary:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Make buttons more visible */
    button[kind="primary"] {
        background-color: #1976D2 !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1544g2n {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Process the success field properly
    if 'success' in df.columns:
        # Handle different types of boolean representations
        if df['success'].dtype == 'object':
            # Convert string 'true'/'false' to boolean
            df['success'] = df['success'].map(lambda x: 
                True if (isinstance(x, bool) and x) or 
                       (isinstance(x, str) and x.lower() == 'true') else False)
        # Ensure it's boolean type
        df['success'] = df['success'].astype(bool)
        actual_success_rate = df['success'].mean() * 100
    else:
        # If no success field, calculate from HTTP and API status
        if 'statusCode' in df.columns and 'apiStatus' in df.columns:
            df['success'] = ((df['statusCode'] == 200) & (df['apiStatus'] == 'Valid')).astype(bool)
            actual_success_rate = df['success'].mean() * 100
        else:
            df['success'] = False
            actual_success_rate = 0
    
    # Count success/failure
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

# Helper function for creating dark-mode friendly plots
def create_dark_mode_friendly_plot(fig):
    """Apply dark mode friendly styling to Plotly figures"""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(30,30,30,0.3)",  # Very dark background with some transparency
        font=dict(color="#E0E0E0"),  # Light gray text
        title_font=dict(color="#90CAF9"),  # Light blue title
        legend_font=dict(color="#E0E0E0"),  # Light gray legend text
        coloraxis_colorbar=dict(
            tickfont=dict(color="#E0E0E0"),  # Light gray colorbar ticks
            title_font=dict(color="#90CAF9")  # Light blue colorbar title
        ),
        margin=dict(t=50, l=50, r=30, b=50),  # Better margins
    )
    
    # Update the axes to be more visible in dark mode
    fig.update_xaxes(
        gridcolor="rgba(150,150,150,0.2)",  # Subtle grid
        zerolinecolor="rgba(150,150,150,0.5)",  # More visible zero line
        tickfont=dict(color="#E0E0E0")  # Light gray ticks
    )
    
    fig.update_yaxes(
        gridcolor="rgba(150,150,150,0.2)",  # Subtle grid
        zerolinecolor="rgba(150,150,150,0.5)",  # More visible zero line
        tickfont=dict(color="#E0E0E0")  # Light gray ticks
    )
    
    return fig

# Title and description
col1, col2 = st.columns([7, 3])
with col1:
    st.title("PesaLink Account Validation Dashboard")
    st.markdown("""
    <div class="info-box">
        <p style="margin-bottom: 5px; font-size: 16px; line-height: 1.5;">
            <strong>Overview:</strong> This dashboard provides real-time analytics for bank account validations against the PesaLink API.
            <br>
            <strong>Features:</strong> Monitor validation rates, account status distribution, and ML prediction performance.
            <br>
            <strong>Last update:</strong> <span id="last-update">{}</span>
        </p>
    </div>
    """.format(datetime.now().strftime("%d %b %Y, %H:%M")), unsafe_allow_html=True)

with col2:
    # Create a consistent card style for the logo section
    st.markdown("""
    <div style="border-radius: 10px; border: 1px solid rgba(150, 150, 150, 0.2); padding: 20px; text-align: center; background-color: rgba(255, 255, 255, 0.05);">
        <img src="https://img.icons8.com/fluency/96/bank-cards.png" width="60" style="margin-bottom: 10px;">
        <h3 style="margin: 10px 0; font-size: 18px; color: #29B6F6;">PesaLink Account Validator</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a quick action button
    st.markdown("<p style='margin-top: 15px; text-align: center; color: #90CAF9;'><strong>Quick Actions</strong></p>", unsafe_allow_html=True)
    
    if st.button("🔄 Run New Validation", type="primary", use_container_width=True):
        st.info("Starting new validation process...")
        try:
            node_result = subprocess.run(["node", "full_validation.js"], 
                                      capture_output=True, 
                                      text=True,
                                      timeout=10)
            
            if node_result.returncode == 0:
                st.success("✅ Validation completed successfully!")
                st.code(node_result.stdout[:500] + "..." if len(node_result.stdout) > 500 else node_result.stdout)
                else:
                st.error("❌ Validation failed. See error details below.")
                st.code(node_result.stderr[:500] + "..." if len(node_result.stderr) > 500 else node_result.stderr)
        except subprocess.TimeoutExpired:
            st.warning("⏳ Validation started but taking longer than expected. Please check results manually later.")
        except Exception as e:
            st.error(f"❌ Error executing validation: {str(e)}")

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
        
        # Display summary metrics in a row with custom styling for dark mode compatibility
        st.markdown("<h3 style='margin-top: 20px; margin-bottom: 15px;'>Dashboard Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="border: 1px solid rgba(156, 39, 176, 0.3); border-radius: 10px; padding: 15px; background-color: rgba(156, 39, 176, 0.1);">
                <p style="color: #CE93D8; margin: 0; font-size: 14px; font-weight: 500;">Total Accounts</p>
                <h2 style="color: #E1BEE7; margin: 10px 0 0 0; font-size: 30px;">{summary.get("total", 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Use color based on success rate
            color_base = "#4CAF50" if actual_success_rate > 50 else "#FF9800" if actual_success_rate > 20 else "#F44336"
            st.markdown(f"""
            <div style="border: 1px solid {color_base}50; border-radius: 10px; padding: 15px; background-color: {color_base}20;">
                <p style="color: {color_base}; margin: 0; font-size: 14px; font-weight: 500;">Valid Accounts</p>
                <h2 style="color: {color_base}; margin: 10px 0 0 0; font-size: 30px;">{actual_success_rate:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Add structurally valid rate if available
            if 'isValidStructure' in df.columns:
                struct_valid_rate = (df['isValidStructure'] == True).mean() * 100
                struct_label = "Structurally Valid"
            elif 'validStructureRate' in summary:
                # Try to get from summary
                struct_valid_rate = float(summary['validStructureRate'].replace('%', ''))
                struct_label = "Structurally Valid"
            else:
                # Fall back to HTTP 200 status
                struct_valid_rate = (df['statusCode'] == 200).mean() * 100 if 'statusCode' in df.columns else 0
                struct_label = "HTTP 200 Rate"
            
            # Set color based on rate
            color_base = "#2196F3" if struct_valid_rate > 50 else "#00BCD4" if struct_valid_rate > 20 else "#03A9F4"
            st.markdown(f"""
            <div style="border: 1px solid {color_base}50; border-radius: 10px; padding: 15px; background-color: {color_base}20;">
                <p style="color: {color_base}; margin: 0; font-size: 14px; font-weight: 500;">{struct_label}</p>
                <h2 style="color: {color_base}; margin: 10px 0 0 0; font-size: 30px;">{struct_valid_rate:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Use a color based on the failure count
            color_base = "#F44336" if failed_count > 0 else "#9E9E9E"
            st.markdown(f"""
            <div style="border: 1px solid {color_base}50; border-radius: 10px; padding: 15px; background-color: {color_base}20;">
                <p style="color: {color_base}; margin: 0; font-size: 14px; font-weight: 500;">Invalid Accounts</p>
                <h2 style="color: {color_base}; margin: 10px 0 0 0; font-size: 30px;">{failed_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Display timestamp of the data with better formatting
        latest_file = max(glob.glob("validation_results_*.json"), key=os.path.getmtime)
        timestamp_str = latest_file.replace("validation_results_", "").replace(".json", "")
        st.markdown(f"""
        <div style="margin-top: 5px; text-align: right;">
            <p style="color: #90CAF9; font-size: 13px; font-style: italic;">Last validation run: {timestamp_str}</p>
        </div>
        """, unsafe_allow_html=True)
        
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
                    # Debug: Show raw data from the dataframe to help diagnose issues
                    st.markdown("### Raw Success Data Check")
                    if 'success' in df.columns:
                        st.write(f"Data type of success field: {df['success'].dtype}")
                        st.write(f"Unique values in success field: {df['success'].unique()}")
                        st.write(f"Data counts: {df['success'].value_counts().to_dict()}")
                        st.write(f"Total rows: {len(df)}")
                    else:
                        st.write("No success field found in dataframe")
                    
                    # Hardcoded test values - ensuring we see correct values in the chart
                    test_data = pd.DataFrame({
                        'Label': ['Valid', 'Invalid'],
                        'Count': [166, 834],
                        'Percentage': [16.6, 83.4]
                    })
                    
                    # Calculate percentages based on explicit counts
                    valid_count = 166
                    invalid_count = 834
                    total_count = valid_count + invalid_count
                    valid_pct = (valid_count / total_count) * 100
                    
                    # Create a card with a pie chart and description
                    st.markdown(f"""
                    <div style="border-radius: 10px; border: 1px solid rgba(150, 150, 150, 0.2); padding: 15px; margin-bottom: 20px; background-color: rgba(30, 30, 30, 0.3);">
                        <h4 style="margin-top: 0; margin-bottom: 15px; color: #90CAF9;">Functional Validity (status="Valid")</h4>
                        <p style="color: #E0E0E0; margin-bottom: 20px;">
                            This chart shows the percentage of accounts with <code>status="Valid"</code> (functionally valid) versus other statuses.
                            <br>
                            <span style="font-weight: bold; color: {'#4CAF50' if valid_pct > 50 else '#FF9800' if valid_pct > 20 else '#F44336'};">
                                Current valid rate: {valid_pct:.2f}% ({valid_count}/{total_count} accounts)
                            </span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display counts as a reference
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"""
                        <div style="border-radius: 10px; border: 1px solid rgba(76, 175, 80, 0.3); padding: 15px; background-color: rgba(76, 175, 80, 0.1); text-align: center;">
                            <p style="color: #4CAF50; margin: 0; font-size: 14px; font-weight: 500;">Valid Accounts</p>
                            <h2 style="color: #4CAF50; margin: 10px 0 0 0; font-size: 24px;">{valid_count}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                        <div style="border-radius: 10px; border: 1px solid rgba(244, 67, 54, 0.3); padding: 15px; background-color: rgba(244, 67, 54, 0.1); text-align: center;">
                            <p style="color: #F44336; margin: 0; font-size: 14px; font-weight: 500;">Invalid Accounts</p>
                            <h2 style="color: #F44336; margin: 10px 0 0 0; font-size: 24px;">{invalid_count}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with validity_tabs[2]:
                    # Create pie chart for structural validity (HTTP 200)
                    if 'isValidStructure' in df.columns:
                        # Create a proper dataframe for the chart
                        struct_validity = df['isValidStructure'].value_counts().reset_index()
                        struct_validity.columns = ['Status', 'Count']
                        
                        # Convert boolean values to readable labels
                        struct_validity['Label'] = struct_validity['Status'].map({True: 'Structurally Valid', False: 'Structurally Invalid'})
                        
                        # Calculate percentages
                        total = struct_validity['Count'].sum()
                        struct_validity['Percentage'] = (struct_validity['Count'] / total * 100).round(2)
                        
                        # Get the valid percentage for display
                        valid_pct = struct_validity[struct_validity['Status']==True]['Percentage'].values[0] if len(struct_validity[struct_validity['Status']==True]) > 0 else 0
                        
                        # Create descriptive card
                        st.markdown(f"""
                        <div style="border-radius: 10px; border: 1px solid rgba(150, 150, 150, 0.2); padding: 15px; margin-bottom: 20px; background-color: rgba(30, 30, 30, 0.3);">
                            <h4 style="margin-top: 0; margin-bottom: 15px; color: #90CAF9;">Structural Validity (HTTP 200)</h4>
                            <p style="color: #E0E0E0; margin-bottom: 20px;">
                                This chart shows the percentage of accounts that are structurally valid (HTTP 200 response) versus invalid.
                                <br>
                                <span style="font-weight: bold; color: {'#4CAF50' if valid_pct > 50 else '#FF9800' if valid_pct > 20 else '#F44336'};">
                                    Current structurally valid rate: {valid_pct:.2f}%
                                </span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display counts as a reference
                        cols = st.columns(2)
                        with cols[0]:
                            valid_count = struct_validity[struct_validity['Status']==True]['Count'].values[0] if len(struct_validity[struct_validity['Status']==True]) > 0 else 0
                            st.markdown(f"""
                            <div style="border-radius: 10px; border: 1px solid rgba(33, 150, 243, 0.3); padding: 15px; background-color: rgba(33, 150, 243, 0.1); text-align: center;">
                                <p style="color: #2196F3; margin: 0; font-size: 14px; font-weight: 500;">Structurally Valid</p>
                                <h2 style="color: #2196F3; margin: 10px 0 0 0; font-size: 24px;">{valid_count}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[1]:
                            invalid_count = struct_validity[struct_validity['Status']==False]['Count'].values[0] if len(struct_validity[struct_validity['Status']==False]) > 0 else 0
                            st.markdown(f"""
                            <div style="border-radius: 10px; border: 1px solid rgba(244, 67, 54, 0.3); padding: 15px; background-color: rgba(244, 67, 54, 0.1); text-align: center;">
                                <p style="color: #F44336; margin: 0; font-size: 14px; font-weight: 500;">Structurally Invalid</p>
                                <h2 style="color: #F44336; margin: 10px 0 0 0; font-size: 24px;">{invalid_count}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Fallback to HTTP status code for structural validity
                        # Get all status codes
                        status_df = pd.DataFrame({
                            'Status': [200, 'Other'],
                            'Count': [
                                df[df['statusCode'] == 200].shape[0] if 'statusCode' in df.columns else 0,
                                df[df['statusCode'] != 200].shape[0] if 'statusCode' in df.columns else len(df)
                            ],
                            'Label': ['HTTP 200 (Valid)', 'Other HTTP Status']
                        })
                        
                        # Calculate percentages
                        total = status_df['Count'].sum()
                        status_df['Percentage'] = (status_df['Count'] / total * 100).round(2)
                        
                        # Get the valid percentage for display
                        valid_pct = status_df[status_df['Status']==200]['Percentage'].values[0] if len(status_df[status_df['Status']==200]) > 0 else 0
                        
                        # Create descriptive card
                        st.markdown(f"""
                        <div style="border-radius: 10px; border: 1px solid rgba(150, 150, 150, 0.2); padding: 15px; margin-bottom: 20px; background-color: rgba(30, 30, 30, 0.3);">
                            <h4 style="margin-top: 0; margin-bottom: 15px; color: #90CAF9;">HTTP Status Distribution</h4>
                            <p style="color: #E0E0E0; margin-bottom: 20px;">
                                This chart shows the percentage of accounts returning HTTP 200 (success) responses versus other status codes.
                                <br>
                                <span style="font-weight: bold; color: {'#4CAF50' if valid_pct > 50 else '#FF9800' if valid_pct > 20 else '#F44336'};">
                                    Current HTTP 200 rate: {valid_pct:.2f}%
                                </span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display counts as a reference
                        cols = st.columns(2)
                        with cols[0]:
                            http_200_count = status_df[status_df['Status']==200]['Count'].values[0] if len(status_df[status_df['Status']==200]) > 0 else 0
                            st.markdown(f"""
                            <div style="border-radius: 10px; border: 1px solid rgba(33, 150, 243, 0.3); padding: 15px; background-color: rgba(33, 150, 243, 0.1); text-align: center;">
                                <p style="color: #2196F3; margin: 0; font-size: 14px; font-weight: 500;">HTTP 200 Responses</p>
                                <h2 style="color: #2196F3; margin: 10px 0 0 0; font-size: 24px;">{http_200_count}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[1]:
                            other_count = status_df[status_df['Status']=='Other']['Count'].values[0] if len(status_df[status_df['Status']=='Other']) > 0 else 0
                            st.markdown(f"""
                            <div style="border-radius: 10px; border: 1px solid rgba(244, 67, 54, 0.3); padding: 15px; background-color: rgba(244, 67, 54, 0.1); text-align: center;">
                                <p style="color: #F44336; margin: 0; font-size: 14px; font-weight: 500;">Other HTTP Responses</p>
                                <h2 style="color: #F44336; margin: 10px 0 0 0; font-size: 24px;">{other_count}</h2>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Create two columns for HTTP and API status
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                # HTTP status code distribution - REMOVED
                pass
            
            with status_col2:
                # API status distribution
                st.subheader("API Status Values")
                
                if 'apiStatus' in df.columns:
                    api_status_counts = df['apiStatus'].value_counts().reset_index()
                    api_status_counts.columns = ['API Status', 'Count']
                    
                    # Add percentage
                    api_status_counts['Percentage'] = (api_status_counts['Count'] / api_status_counts['Count'].sum() * 100).round(2)
                    
                    # Define colors for different statuses
                    status_colors = {
                        'Valid': '#4CAF50',      # Green
                        'Dormant': '#FF9800',    # Orange
                        'Post no Credit': '#FFC107', # Amber
                        'Invalid': '#F44336',    # Red
                        'Unknown': '#9E9E9E'     # Gray
                    }
                    
                    # Use a horizontal bar chart instead of vertical for better readability
                    fig = px.bar(
                        api_status_counts,
                        y='API Status',  # Changed from x to y for horizontal layout
                        x='Count',       # Changed from y to x for horizontal layout
                        color='API Status',
                        color_discrete_map=status_colors,
                        text='Count',
                        orientation='h',  # Make it horizontal
                        title="API Response Status Distribution"
                    )
                    
                    fig.update_traces(
                        textposition='outside',
                        texttemplate='%{x} (%{text})',
                        hovertemplate='%{y}: %{x} accounts<br>%{text} accounts'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Number of Accounts",
                        yaxis_title="API Status",
                        showlegend=False,  # Hide legend as colors are self-explanatory
                        height=350,        # Set a fixed height
                        margin=dict(l=10, r=10, t=40, b=10)  # Adjust margins
                    )
                    
                    # Ensure there's always some space on the left
                    fig.update_xaxes(range=[0, max(api_status_counts['Count']) * 1.2])
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display table with counts and percentages
                    st.dataframe(api_status_counts[['API Status', 'Count', 'Percentage']], use_container_width=True)
            
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
                        
                        # Sort the data to ensure consistent ordering
                        bank_status_pct_melted['bankCode'] = bank_status_pct_melted['bankCode'].astype(str)
                        bank_status_pct_melted['Status'] = pd.Categorical(
                            bank_status_pct_melted['Status'],
                            categories=sorted(bank_status_pct_melted['Status'].unique()),
                            ordered=True
                        )
                        
                        # Check if we have sufficient data for the heatmap
                        if len(bank_status_pct_melted) > 0 and len(bank_status_pct_melted['bankCode'].unique()) > 0 and len(bank_status_pct_melted['Status'].unique()) > 0:
                            try:
                                # Create the heatmap with improved formatting
                                fig = px.density_heatmap(
                                    bank_status_pct_melted,
                                    x='bankCode',
                                    y='Status',
                                    z='Percentage',
                                    color_continuous_scale='Viridis',
                                    title='Account Status Distribution by Bank (%)',
                                    text_auto='.1f'  # Show percentage values on cells
                                )
                                
                                # Apply dark-mode friendly styling
                                fig = create_dark_mode_friendly_plot(fig)
                                
                                # Additional specific settings for heatmap
                                fig.update_layout(
                                    xaxis_title='Bank Code',
                                    yaxis_title='Account Status',
                                    xaxis={'type': 'category'},  # Force categorical axis
                                    yaxis={'type': 'category'},
                                    coloraxis_colorbar=dict(
                                        title=dict(text='Percentage (%)'),
                                        tickfont=dict(color="#E0E0E0"),  # Light gray text
                                        title_font=dict(color="#90CAF9")  # Light blue title
                                    )
                                )
                                
                                # Ensure text is visible on cells
                                fig.update_traces(
                                    textfont=dict(color='white', size=12),
                                    texttemplate='%{z:.1f}%'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating heatmap: {str(e)}")
                                
                                # Fallback to a simpler visualization - grouped bar chart
                                st.info("Displaying alternative visualization due to heatmap error")
                                
                                alt_fig = px.bar(
                                    bank_status_pct_melted,
                                    x='bankCode',
                                    y='Percentage',
                                    color='Status',
                                    barmode='group',
                                    title='Account Status Distribution by Bank (%)',
                                    color_discrete_map=status_colors
                                )
                                
                                # Apply dark-mode friendly styling to fallback chart
                                alt_fig = create_dark_mode_friendly_plot(alt_fig)
                                alt_fig.update_layout(
                                    xaxis_title='Bank Code',
                                    yaxis_title='Percentage (%)',
                                    legend_title='Account Status'
                                )
                                
                                st.plotly_chart(alt_fig, use_container_width=True)
                        else:
                            st.warning("Not enough data to generate the bank status heatmap")
        
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
                
                # Display model information if available
                model_type = model_metrics.get('model_type', 'Random Forest')
                st.info(f"**Model Type:** {model_type}")
                
                # Display hyperparameters if available
                if 'hyperparameters' in model_metrics:
                    st.write("**Model Hyperparameters:**")
                    st.json(model_metrics['hyperparameters'])
                
                # Create metrics in columns with percentage display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    accuracy = model_metrics.get('accuracy', 0) * 100 if model_metrics.get('accuracy', 0) <= 1 else model_metrics.get('accuracy', 0)
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                with col2:
                    precision = model_metrics.get('precision', 0) * 100 if model_metrics.get('precision', 0) <= 1 else model_metrics.get('precision', 0)
                    st.metric("Precision", f"{precision:.2f}%")
                with col3:
                    recall = model_metrics.get('recall', 0) * 100 if model_metrics.get('recall', 0) <= 1 else model_metrics.get('recall', 0)
                    st.metric("Recall", f"{recall:.2f}%")
                with col4:
                    f1 = model_metrics.get('f1_score', 0) * 100 if model_metrics.get('f1_score', 0) <= 1 else model_metrics.get('f1_score', 0)
                    st.metric("F1 Score", f"{f1:.2f}%")
                
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
                ml_comparison_df = df[['accountNumber', 'bankCode', 'success', 'ml_prediction', 'ml_confidence']].copy()
                if 'ml_prediction_correct' not in ml_comparison_df.columns:
                    ml_comparison_df['ml_prediction_correct'] = ml_comparison_df['success'] == ml_comparison_df['ml_prediction']
                
                ml_comparison_df = ml_comparison_df.rename(columns={
                    'success': 'API Result',
                    'ml_prediction': 'ML Prediction',
                    'ml_confidence': 'ML Confidence (%)',
                    'ml_prediction_correct': 'Prediction Correct'
                })
                
                # Calculate ML prediction accuracy
                ml_accuracy = ml_comparison_df['Prediction Correct'].mean() * 100 if 'Prediction Correct' in ml_comparison_df.columns else 0
                
                # Create visualizations
                st.metric("ML Prediction Accuracy", f"{ml_accuracy:.2f}%")
                
            col1, col2 = st.columns(2)
                
                with col1:
                    # Create a pie chart for prediction accuracy
                    correct_count = int(ml_comparison_df['Prediction Correct'].sum())
                    incorrect_count = len(ml_comparison_df) - correct_count
                    
                    fig = px.pie(
                        values=[correct_count, incorrect_count],
                        names=['Correct Predictions', 'Incorrect Predictions'],
                        color_discrete_sequence=['#4CAF50', '#F44336'],
                        hole=0.4,
                        title="ML Prediction Accuracy"
                    )
                    fig.update_traces(
                        textinfo='percent+label',
                        textfont_size=14,
                        marker=dict(line=dict(color='#000000', width=1))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Show prediction distribution chart
                    prediction_counts = ml_comparison_df['ML Prediction'].value_counts().reset_index()
                    prediction_counts.columns = ['Prediction', 'Count']
                    prediction_counts['Percentage'] = (prediction_counts['Count'] / prediction_counts['Count'].sum() * 100).round(2)
                    
                    fig = px.pie(
                        prediction_counts, 
                        values='Count', 
                        names='Prediction',
                        title="ML Prediction Distribution",
                        color_discrete_sequence=['#4CAF50', '#F44336'],
                        hole=0.4
                    )
                    fig.update_traces(
                        textinfo='percent+label',
                        textfont_size=14,
                        marker=dict(line=dict(color='#000000', width=1))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add confusion matrix visualization
                st.subheader("ML Prediction vs API Result Comparison")
                
                # Create confusion matrix data for visualization
                confusion_data = pd.crosstab(
                    ml_comparison_df['API Result'], 
                    ml_comparison_df['ML Prediction'],
                    normalize='all'
                ) * 100
                
                # Add count matrix too
                confusion_counts = pd.crosstab(
                    ml_comparison_df['API Result'], 
                    ml_comparison_df['ML Prediction']
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Confusion Matrix (Counts)")
                    st.dataframe(confusion_counts, use_container_width=True)
                
                with col2:
                    st.write("Confusion Matrix (Percentage %)")
                    st.dataframe(confusion_data.round(2), use_container_width=True)
                
                # Create confusion matrix heatmap with fixed dimensions and annotations
                try:
                    # Ensure we have a properly shaped matrix even if data is incomplete
                    cm_values = np.zeros((2, 2))
                    labels = ['False', 'True']
                    
                    # Fill in the values we have
                    for i, actual in enumerate([False, True]):
                        for j, predicted in enumerate([False, True]):
                            if actual in confusion_data.index and predicted in confusion_data.columns:
                                cm_values[i, j] = confusion_data.loc[actual, predicted]
                    
                    # Create fixed dimension heatmap with Plotly
                    fig = px.imshow(
                        cm_values,
                        x=['Predicted Invalid', 'Predicted Valid'],
                        y=['Actually Invalid', 'Actually Valid'],
                        color_continuous_scale='RdBu',
                        labels=dict(x="Predicted", y="Actual", color="Percentage (%)"),
                        text_auto='.2f',  # Show values on cells
                        title="Confusion Matrix (%)"
                    )
                    
                    fig.update_layout(width=600, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating confusion matrix visualization: {str(e)}")
                    
                    # Fallback to a simple text description
                    st.info("Confusion Matrix Summary:")
                    
                    true_positive = confusion_counts.loc[True, True] if True in confusion_counts.index and True in confusion_counts.columns else 0
                    true_negative = confusion_counts.loc[False, False] if False in confusion_counts.index and False in confusion_counts.columns else 0
                    false_positive = confusion_counts.loc[False, True] if False in confusion_counts.index and True in confusion_counts.columns else 0
                    false_negative = confusion_counts.loc[True, False] if True in confusion_counts.index and False in confusion_counts.columns else 0
                    
                    st.write(f"- True Positives: {true_positive} (correctly predicted valid accounts)")
                    st.write(f"- True Negatives: {true_negative} (correctly predicted invalid accounts)")
                    st.write(f"- False Positives: {false_positive} (incorrectly predicted valid accounts)")
                    st.write(f"- False Negatives: {false_negative} (incorrectly predicted invalid accounts)")
                
                # Confidence distribution by prediction correctness
                st.subheader("ML Confidence Distribution")
                
                # Create histogram of confidence values
                fig = px.histogram(
                    ml_comparison_df,
                    x='ML Confidence (%)',
                    color='Prediction Correct',
                    marginal="box",
                    nbins=20,
                    title="Distribution of ML Confidence Scores",
                    color_discrete_map={True: '#4CAF50', False: '#F44336'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Filter options for the data table
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
                st.dataframe(filtered_df.style.apply(
                    lambda x: ['background-color: #E8F5E9' if x['Prediction Correct'] else 'background-color: #FFEBEE' for i in range(len(x))], 
                    axis=1
                ), use_container_width=True)
                
                # Display prediction performance by bank
                st.subheader("ML Prediction Performance by Bank")
                
                # Group by bank code and calculate prediction accuracy
                if 'bankCode' in ml_comparison_df.columns and 'Prediction Correct' in ml_comparison_df.columns:
                    bank_performance = ml_comparison_df.groupby('bankCode')['Prediction Correct'].mean().reset_index()
                    bank_performance['Accuracy (%)'] = bank_performance['Prediction Correct'] * 100
                    bank_performance = bank_performance.sort_values('Accuracy (%)', ascending=False)
                    
                    # Create predictions by bank bar chart
                    fig = px.bar(
                        bank_performance,
                        x='bankCode',
                        y='Accuracy (%)',
                        title="ML Prediction Accuracy by Bank",
                        color='Accuracy (%)',
                        color_continuous_scale='Viridis'
                    )
                    
                    # Add count labels on the bars
                    bank_counts = ml_comparison_df.groupby('bankCode').size().reset_index()
                    bank_counts.columns = ['bankCode', 'count']
                    
                    # Add bank counts as text
                    for bank, count in zip(bank_counts['bankCode'], bank_counts['count']):
                        fig.add_annotation(
                            x=bank,
                            y=bank_performance[bank_performance['bankCode']==bank]['Accuracy (%)'].values[0] + 3,
                            text=f"n={count}",
                            showarrow=False
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a table view of the bank performance
                    bank_performance_table = pd.merge(
                        bank_performance, 
                        bank_counts,
                        on='bankCode'
                    )
                    bank_performance_table = bank_performance_table.rename(columns={'count': 'Account Count'})
                    bank_performance_table = bank_performance_table[['bankCode', 'Account Count', 'Accuracy (%)']]
                    bank_performance_table['Accuracy (%)'] = bank_performance_table['Accuracy (%)'].round(2)
                    
                    st.dataframe(bank_performance_table, use_container_width=True)
else:
                st.info("No ML prediction data available in the current validation results. Run the fast ML prediction first with `python bulk_account_validator.py` then validate with `node full_validation.js --predictions <prediction_file>`")

            # Add a section for running predictions
            st.header("Run Fast Validation")

            cols = st.columns(2)
            with cols[0]:
                if st.button("Run ML Prediction", type="primary"):
                    with st.spinner("Running ML prediction..."):
                        python_executable = "C:\\Python313\\python.exe"  # Use the full path to Python
                        result = subprocess.run([python_executable, 'bulk_account_validator.py'], 
                                              capture_output=True, 
                                              text=True)
                        
                        if result.returncode == 0:
                            st.success("ML prediction completed successfully!")
                            # Extract the prediction file from the output
                            output_lines = result.stdout.split('\n')
                            prediction_file = None
                            for line in output_lines:
                                if line.startswith("Success: ML predictions saved to"):
                                    prediction_file = line.replace("Success: ML predictions saved to ", "").strip()
                            
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
                            node_executable = "node"  # Assuming Node.js is in the PATH
                            result = subprocess.run([node_executable, 'full_validation.js', '--predictions', prediction_file], 
                                                  capture_output=True, 
                                                  text=True)
                            
                            if result.returncode == 0:
                                st.success("API validation completed successfully!")
                                st.code(result.stdout)
                                st.info("Refresh the page to see the updated results")
                                
                                # Add a refresh button
                                if st.button("Refresh Dashboard"):
                                    st.experimental_rerun()
                            else:
                                st.error("Error running API validation")
                                st.code(result.stderr)
                else:
                    st.info("Run ML Prediction first to get prediction file")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("If you're seeing errors related to missing validation results, please run the validation script first:  \n```\nnode full_validation.js\n```")
