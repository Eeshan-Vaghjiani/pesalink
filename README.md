# PesaLink Account Validation

This project contains scripts to validate bank accounts using the PesaLink Account Validation Service API.

## Files

- `sample_1000_accounts.csv` - Sample data containing 1000 bank accounts with their account numbers and bank codes
- `validate_account.js` - Basic script to test the API with a single account
- `validate_account_fetch.js` - Script to test different authentication methods for the API
- `validate_bulk.js` - Script to validate multiple accounts in batches
- `full_validation.js` - Comprehensive script to validate all accounts with detailed reporting

## How to Use

### Install Dependencies

```bash
npm install node-fetch@2 csv-parse
```

### Running the Full Validation

To validate all 1000 accounts in the sample file:

```bash
node full_validation.js
```

The script will:
1. Process accounts in batches of 10
2. Add delays between requests to avoid rate limiting
3. Generate detailed reports of the validation results

### API Authentication

The API requires a Bearer token authentication. Use the API key as follows:

```js
headers: {
  'Authorization': `Bearer ${API_KEY}`
}
```

## Output Files

The validation script generates two output files:

1. `validation_results_[timestamp].json` - Contains detailed validation results for each account
2. `validation_summary_[timestamp].json` - Contains summary statistics about the validation process

## API Documentation

The Account Validation Service API is available at:
https://account-validation-service.dev.pesalink.co.ke/api-docs/

### Endpoints

- `POST /api/validate` - Validate an account using account number and bank code
- `GET /download/{fileName}` - Download a CSV file of accounts data
- `GET /api/key` - Retrieve the API Key 