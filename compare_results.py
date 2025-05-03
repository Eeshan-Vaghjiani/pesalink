import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import os
import sys
import seaborn as sns

def load_expected_results(file_path):
    """Load the expected validity results from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Make sure required columns exist
        if 'Account Number' not in df.columns or 'Expected Validity' not in df.columns:
            print(f"Error: Required columns missing from {file_path}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading expected results: {str(e)}")
        return None

def load_actual_results(file_path):
    """Load the actual validation results from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        if 'accountNumber' not in df.columns:
            print(f"Error: Required columns missing from {file_path}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading actual results: {str(e)}")
        return None

def compare_results(expected_df, actual_df):
    """Compare expected vs actual validity results"""
    # Rename columns for consistency
    expected_df = expected_df.rename(columns={
        'Account Number': 'accountNumber',
        'Bank Code': 'bankCode',
        'Expected Validity': 'expectedValidity',
        'Expected Status': 'expectedStatus'
    })
    
    # Convert expected validity to boolean
    expected_df['expectedValid'] = expected_df['expectedValidity'] == 'Valid'
    
    # Merge datasets on account number and bank code
    merged = pd.merge(
        expected_df,
        actual_df,
        on=['accountNumber', 'bankCode'],
        how='left'
    )
    
    # If success column doesn't exist, try to determine from apiStatus
    if 'success' not in merged.columns and 'apiStatus' in merged.columns:
        merged['success'] = merged['apiStatus'] == 'Valid'
    
    # Handle missing values (accounts not found in validation results)
    if merged['success'].isna().any():
        print(f"Warning: {merged['success'].isna().sum()} accounts were not found in validation results")
        merged.loc[merged['success'].isna(), 'success'] = False
    
    # Compare expected vs actual
    merged['correct_validity'] = merged['expectedValid'] == merged['success']
    
    # Compare expected status vs actual API status if available
    if 'expectedStatus' in merged.columns and 'apiStatus' in merged.columns:
        merged['correct_status'] = merged['expectedStatus'] == merged['apiStatus']
    else:
        merged['correct_status'] = None
    
    # Calculate accuracy metrics for validity
    total_accounts = len(merged)
    correct_predictions = merged['correct_validity'].sum()
    accuracy = correct_predictions / total_accounts
    
    # Calculate confusion matrix values for validity
    true_positives = ((merged['expectedValid'] == True) & (merged['success'] == True)).sum()
    true_negatives = ((merged['expectedValid'] == False) & (merged['success'] == False)).sum()
    false_positives = ((merged['expectedValid'] == False) & (merged['success'] == True)).sum()
    false_negatives = ((merged['expectedValid'] == True) & (merged['success'] == False)).sum()
    
    # Calculate precision, recall and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate status accuracy if available
    if 'correct_status' in merged.columns and not merged['correct_status'].isna().all():
        status_accuracy = merged['correct_status'].mean()
        status_match_count = merged['correct_status'].sum()
    else:
        status_accuracy = None
        status_match_count = 0
    
    # Return results
    return {
        'merged_data': merged,
        'total_accounts': total_accounts,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'confusion_matrix': {
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'status_accuracy': status_accuracy,
        'status_match_count': status_match_count
    }

def generate_report(results, output_file=None):
    """Generate a detailed report of the comparison results"""
    merged_data = results['merged_data']
    cm = results['confusion_matrix']
    
    # Print summary metrics
    print("\n=== Validation Results Comparison ===")
    print(f"Total Accounts: {results['total_accounts']}")
    print(f"Correct Predictions: {results['correct_predictions']} ({results['accuracy']*100:.2f}%)")
    print(f"Incorrect Predictions: {results['total_accounts'] - results['correct_predictions']} ({(1-results['accuracy'])*100:.2f}%)")
    
    # Print status accuracy if available
    if results['status_accuracy'] is not None:
        print(f"\nAPI Status Accuracy: {results['status_accuracy']*100:.2f}%")
        print(f"Status Matches: {results['status_match_count']} of {results['total_accounts']}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Positives (Expected Valid, Actual Valid): {cm['true_positives']}")
    print(f"True Negatives (Expected Invalid, Actual Invalid): {cm['true_negatives']}")
    print(f"False Positives (Expected Invalid, Actual Valid): {cm['false_positives']}")
    print(f"False Negatives (Expected Valid, Actual Invalid): {cm['false_negatives']}")
    
    # Print precision, recall, F1
    print("\nPerformance Metrics:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    # Analysis of mismatches
    mismatches = merged_data[merged_data['correct_validity'] == False]
    
    # If there are Invalid Type codes, analyze by type
    if 'Invalid Type' in merged_data.columns and not mismatches.empty:
        print("\nMismatches by Invalid Type:")
        invalid_type_counts = mismatches['Invalid Type'].value_counts()
        for invalid_type, count in invalid_type_counts.items():
            print(f"- {invalid_type}: {count} mismatches")
    
    # If there are API status values, analyze mismatches by status
    if 'apiStatus' in merged_data.columns and not mismatches.empty:
        print("\nMismatches by API Status:")
        api_status_counts = mismatches['apiStatus'].value_counts()
        for status, count in api_status_counts.items():
            print(f"- {status}: {count} mismatches")
    
    # Add status comparison analysis if available
    if 'correct_status' in merged_data.columns and not merged_data['correct_status'].isna().all():
        status_mismatches = merged_data[merged_data['correct_status'] == False]
        
        if not status_mismatches.empty:
            print("\nAPI Status Mismatches:")
            status_crosstab = pd.crosstab(
                status_mismatches['expectedStatus'], 
                status_mismatches['apiStatus'], 
                margins=True, 
                margins_name="Total"
            )
            print(status_crosstab)
    
    # If output file specified, save detailed mismatch data
    if output_file and not mismatches.empty:
        mismatches.to_csv(output_file, index=False)
        print(f"\nDetailed mismatches saved to {output_file}")
    
    # Return results for potential further analysis
    return results

def plot_confusion_matrix(results, output_file=None):
    """Plot a visual confusion matrix"""
    merged_data = results['merged_data']
    cm = results['confusion_matrix']
    
    # Create figure for validity confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Create the confusion matrix data
    cm_data = [
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ]
    
    # Plot heatmap
    plt.imshow(cm_data, cmap='Blues')
    
    # Add labels and values
    class_labels = ['Invalid', 'Valid']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.title('Validity Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm_data[i][j]), 
                     ha="center", va="center", 
                     color="white" if cm_data[i][j] > results['total_accounts']/4 else "black")
    
    plt.colorbar(label='Count')
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_file}")
    
    plt.tight_layout()
    plt.show()

    # Plot API status confusion matrix if available
    if ('expectedStatus' in merged_data.columns and 'apiStatus' in merged_data.columns 
        and not merged_data['correct_status'].isna().all()):
        
        # Get unique status values
        expected_statuses = sorted(merged_data['expectedStatus'].unique())
        actual_statuses = sorted(merged_data['apiStatus'].unique())
        
        # Create a status confusion matrix
        status_cm = pd.crosstab(
            merged_data['expectedStatus'], 
            merged_data['apiStatus']
        )
        
        # Create a new figure for status confusion matrix
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns_heatmap = plt.imshow(status_cm, cmap='YlGnBu')
        
        # Add labels
        plt.colorbar(label='Count')
        plt.title('API Status Confusion Matrix')
        plt.xlabel('Actual API Status')
        plt.ylabel('Expected API Status')
        
        # Set x and y ticks
        plt.xticks(range(len(actual_statuses)), actual_statuses, rotation=45)
        plt.yticks(range(len(expected_statuses)), expected_statuses)
        
        # Add count annotations
        for i in range(len(expected_statuses)):
            for j in range(len(actual_statuses)):
                value = status_cm.iloc[i, j] if i < status_cm.shape[0] and j < status_cm.shape[1] else 0
                plt.text(j, i, str(value), 
                         ha="center", va="center", 
                         color="white" if value > results['total_accounts']/10 else "black")
        
        # Save status confusion matrix if output file specified
        if output_file:
            status_cm_file = output_file.replace('.png', '_status.png')
            plt.savefig(status_cm_file, dpi=300, bbox_inches='tight')
            print(f"API Status confusion matrix saved to {status_cm_file}")
        
        plt.tight_layout()
        plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare expected vs actual account validation results')
    parser.add_argument('--expected', required=True, help='Path to CSV file with expected validity results')
    parser.add_argument('--actual', required=True, help='Path to JSON file with actual validation results')
    parser.add_argument('--output', help='Path to save detailed mismatch results', default=None)
    parser.add_argument('--plot', help='Path to save confusion matrix plot', default=None)
    
    args = parser.parse_args()
    
    # Load data
    expected_df = load_expected_results(args.expected)
    if expected_df is None:
        sys.exit(1)
        
    actual_df = load_actual_results(args.actual)
    if actual_df is None:
        sys.exit(1)
    
    # Compare results
    results = compare_results(expected_df, actual_df)
    
    # Generate report
    generate_report(results, args.output)
    
    # Plot confusion matrix if matplotlib is available
    try:
        plot_confusion_matrix(results, args.plot)
    except Exception as e:
        print(f"Error creating confusion matrix plot: {str(e)}")

if __name__ == "__main__":
    main() 