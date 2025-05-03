import pandas as pd
import numpy as np
import random
import csv
import os
from datetime import datetime

def generate_synthetic_accounts(num_accounts=1000, invalid_ratio=0.3):
    """
    Generate a dataset with a mix of valid and invalid accounts for testing
    
    Parameters:
    - num_accounts: Total number of accounts to generate
    - invalid_ratio: Proportion of accounts that should be invalid
    """
    print(f"Generating {num_accounts} synthetic accounts ({invalid_ratio*100:.0f}% invalid)...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Calculate number of accounts for each validity category
    num_invalid = int(num_accounts * invalid_ratio)
    num_valid = num_accounts - num_invalid
    
    # Bank codes - use realistic values
    bank_codes = ['0011', '0023', '0063', '0012', '0072', '0031', '0087', '0041', '0054', '0076']
    
    # Create valid accounts with correct structure (10-12 digits)
    valid_accounts = []
    for i in range(num_valid):
        bank_code = random.choice(bank_codes)
        
        # Valid accounts will have correct number of digits
        account_length = random.choice([10, 11, 12])
        account_number = ''.join([str(random.randint(0, 9)) for _ in range(account_length)])
        
        valid_accounts.append({
            'Account Number': account_number,
            'Bank Code': bank_code,
            'Expected Validity': 'Valid',
            'Expected Status': 'Valid'
        })
    
    # Create invalid accounts with various issues and API status types
    invalid_accounts = []
    for i in range(num_invalid):
        bank_code = random.choice(bank_codes)
        
        # Assign an API status and invalid type
        # Use all possible API status values: Invalid, Dormant, Post no Credit
        api_status = random.choice(['Invalid', 'Dormant', 'Post no Credit'])
        
        # Determine invalid type based on API status
        if api_status == 'Invalid':
            invalid_type = random.choice(['wrong_length', 'non_numeric', 'invalid_bank', 'invalid_account'])
        elif api_status == 'Dormant':
            invalid_type = 'dormant_account'
        else:  # Post no Credit
            invalid_type = 'post_no_credit'
            
        # Generate account number based on invalid type
        if invalid_type == 'wrong_length':
            # Too short or too long
            account_length = random.choice([8, 9, 13, 14])
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(account_length)])
            
        elif invalid_type == 'non_numeric':
            # Contains non-numeric characters
            account_length = random.choice([10, 11, 12])
            chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            account_number = ''.join([random.choice(chars) for _ in range(account_length)])
            
        elif invalid_type == 'invalid_bank':
            # Valid account number but invalid bank
            account_length = random.choice([10, 11, 12])
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(account_length)])
            bank_code = str(random.randint(9000, 9999))  # Non-existent bank code
            
        else:  # dormant_account, post_no_credit, invalid_account
            # Structurally valid but with special status
            account_length = random.choice([10, 11, 12])
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(account_length)])
            
        invalid_accounts.append({
            'Account Number': account_number,
            'Bank Code': bank_code,
            'Expected Validity': 'Invalid',
            'Expected Status': api_status,
            'Invalid Type': invalid_type
        })
    
    # Combine and shuffle
    all_accounts = valid_accounts + invalid_accounts
    random.shuffle(all_accounts)
    
    return all_accounts

def save_accounts_to_csv(accounts, filename=None):
    """Save the generated accounts to a CSV file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_accounts_{timestamp}.csv"
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Account Number', 'Bank Code', 'Expected Validity']
        # Add Invalid Type if it exists
        if 'Invalid Type' in accounts[0]:
            fieldnames.append('Invalid Type')
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for account in accounts:
            # Only include Expected Validity and Invalid Type in the field info dict
            # Don't include them in the actual CSV
            info = {k: account[k] for k in fieldnames if k not in ['Expected Validity', 'Invalid Type']}
            writer.writerow(info)
    
    print(f"Saved {len(accounts)} accounts to {filename}")
    
    # Also save a version with the expected validity for analysis
    analysis_filename = filename.replace('.csv', '_with_analysis.csv')
    with open(analysis_filename, 'w', newline='') as csvfile:
        fieldnames = ['Account Number', 'Bank Code', 'Expected Validity', 'Expected Status']
        if any('Invalid Type' in account for account in accounts):
            fieldnames.append('Invalid Type')
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accounts)
    
    print(f"Saved analysis version to {analysis_filename}")
    
    return filename, analysis_filename

def main():
    # Generate accounts with 30% invalid by default
    accounts = generate_synthetic_accounts(num_accounts=1000, invalid_ratio=0.3)
    
    # Count by expected validity
    valid_count = sum(1 for acc in accounts if acc['Expected Validity'] == 'Valid')
    invalid_count = sum(1 for acc in accounts if acc['Expected Validity'] == 'Invalid')
    
    print(f"\nGenerated Accounts Summary:")
    print(f"Total Accounts: {len(accounts)}")
    print(f"Expected Valid: {valid_count} ({valid_count/len(accounts)*100:.1f}%)")
    print(f"Expected Invalid: {invalid_count} ({invalid_count/len(accounts)*100:.1f}%)")
    
    # If invalid accounts exist, show breakdown by type and expected status
    if invalid_count > 0:
        # Count by invalid type
        invalid_types = {}
        for acc in accounts:
            if acc['Expected Validity'] == 'Invalid':
                invalid_type = acc.get('Invalid Type', 'unknown')
                invalid_types[invalid_type] = invalid_types.get(invalid_type, 0) + 1
        
        print("\nInvalid Account Types:")
        for type_name, count in invalid_types.items():
            print(f"- {type_name}: {count} ({count/invalid_count*100:.1f}%)")
        
        # Count by expected API status
        status_counts = {}
        for acc in accounts:
            status = acc.get('Expected Status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nExpected API Status Distribution:")
        for status, count in status_counts.items():
            print(f"- {status}: {count} ({count/len(accounts)*100:.1f}%)")
    
    # Save to CSV
    filename, analysis_filename = save_accounts_to_csv(accounts)
    
    print("\nTo validate these accounts, run:")
    print(f"node full_validation.js # (after copying {filename} to sample_1000_accounts.csv)")
    print("\nTo compare the validation results with expected results:")
    print(f"python compare_results.py --expected {analysis_filename} --actual <validation_results_file>")

if __name__ == "__main__":
    main() 