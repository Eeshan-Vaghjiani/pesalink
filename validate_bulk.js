const fs = require('fs');
const fetch = require('node-fetch');
const { parse } = require('csv-parse/sync');

// API Configuration
const API_KEY = '609dab841280674f1a780272f59e9e4e';
const API_URL = 'https://account-validation-service.dev.pesalink.co.ke/api/validate';

// Debugging: Print information about a failed request
function printRequestError(accountNumber, bankCode, response) {
    console.error(`Request failed for account: ${accountNumber}, bank code: ${bankCode}`);
    console.error(`Status: ${response.status}, Status Text: ${response.statusText}`);
    console.error('Request headers:', response.headers);
}

// Function to validate a single account
async function validateAccount(accountNumber, bankCode) {
    console.log(`Validating: Account=${accountNumber}, Bank=${bankCode}`);

    try {
        const body = JSON.stringify({
            accountNumber: accountNumber,
            bankCode: bankCode
        });

        console.log(`Request body: ${body}`);

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: body
        });

        const status = response.status;
        let data;

        try {
            const text = await response.text();
            console.log(`Raw response: ${text}`);
            data = text ? JSON.parse(text) : {};
        } catch (e) {
            data = { error: 'Invalid JSON response' };
        }

        if (status !== 200) {
            printRequestError(accountNumber, bankCode, response);
        }

        return {
            accountNumber,
            bankCode,
            statusCode: status,
            success: status === 200,
            data
        };
    } catch (error) {
        console.error(`Error validating account ${accountNumber}:`, error);
        return {
            accountNumber,
            bankCode,
            success: false,
            error: error.message
        };
    }
}

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Process accounts in batches
async function processBatch(accounts, batchSize = 5, delayMs = 1000) {
    const results = [];

    for (let i = 0; i < accounts.length; i += batchSize) {
        const batch = accounts.slice(i, i + batchSize);
        console.log(`Processing batch ${i / batchSize + 1}/${Math.ceil(accounts.length / batchSize)} (${i + 1}-${Math.min(i + batchSize, accounts.length)} of ${accounts.length} accounts)`);

        // Process one account at a time for better debugging
        for (const account of batch) {
            const result = await validateAccount(account['Account Number'], account['Bank Code']);
            results.push(result);
        }

        // Print summary of this batch
        const batchResults = batch.map((_, index) => results[i + index]);
        const successes = batchResults.filter(r => r.success).length;
        console.log(`Batch results: ${successes}/${batchResults.length} successful validations`);

        // Add delay between batches
        if (i + batchSize < accounts.length) {
            console.log(`Waiting ${delayMs}ms before next batch...`);
            await sleep(delayMs);
        }
    }

    return results;
}

// Main function
async function main() {
    try {
        // Read the CSV file
        console.log('Reading CSV file...');
        const csvData = fs.readFileSync('sample_1000_accounts.csv', 'utf8');

        // Parse CSV
        const records = parse(csvData, {
            columns: true,
            skip_empty_lines: true
        });

        console.log(`Loaded ${records.length} accounts from CSV`);
        console.log('Sample CSV record:', records[0]);

        // Process accounts in batches
        const batchSize = 3;
        const maxAccounts = 5; // Limit for testing purposes
        const accountsToProcess = records.slice(0, maxAccounts);

        console.log(`Processing ${accountsToProcess.length} accounts in batches of ${batchSize}...`);
        const results = await processBatch(accountsToProcess, batchSize);

        // Analyze results
        const successful = results.filter(r => r.success);
        const failed = results.filter(r => !r.success);

        console.log(`\nValidation summary:`);
        console.log(`- Total accounts processed: ${results.length}`);
        console.log(`- Successful validations: ${successful.length}`);
        console.log(`- Failed validations: ${failed.length}`);

        // Save results to file
        const outputFile = 'validation_results.json';
        fs.writeFileSync(outputFile, JSON.stringify(results, null, 2));
        console.log(`\nResults saved to ${outputFile}`);

        // If there are successful validations, show a sample
        if (successful.length > 0) {
            console.log(`\nSample successful validation:`);
            console.log(JSON.stringify(successful[0], null, 2));
        }

        // If there are failures, show a sample
        if (failed.length > 0) {
            console.log(`\nSample failed validation:`);
            console.log(JSON.stringify(failed[0], null, 2));
        }
    } catch (error) {
        console.error('Error in main function:', error);
    }
}

main(); 