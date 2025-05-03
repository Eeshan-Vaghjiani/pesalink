const fs = require('fs');
const fetch = require('node-fetch');
const { parse } = require('csv-parse/sync');
const crypto = require('crypto');

// API Configuration
const API_KEY = '609dab841280674f1a780272f59e9e4e';
const API_URL = 'https://account-validation-service.dev.pesalink.co.ke/api/validate';

// Setup logging
const LOG_DIR = './logs';
const REQUEST_LOG_FILE = `${LOG_DIR}/requests_responses.log`;
const ERROR_LOG_FILE = `${LOG_DIR}/errors.log`;

// Process command line arguments
const args = process.argv.slice(2);
const predictionFileArg = args.find(arg => arg.startsWith('--predictions='));
const predictionFileArgIndex = args.indexOf('--predictions');
let predictionFile = null;

if (predictionFileArg) {
    predictionFile = predictionFileArg.split('=')[1];
} else if (predictionFileArgIndex >= 0 && args.length > predictionFileArgIndex + 1) {
    predictionFile = args[predictionFileArgIndex + 1];
}

// Ensure log directory exists
if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
}

// Helper for logging
function logRequest(accountNumber, bankCode, response, data) {
    const timestamp = new Date().toISOString();

    // Hash account holder name if present for privacy
    let logData = data;
    if (data && data.accountHolderName) {
        logData = { ...data };
        logData.accountHolderName = hashSensitiveData(data.accountHolderName);
    }

    const logEntry = {
        timestamp,
        request: { accountNumber, bankCode },
        statusCode: response.status,
        responseData: logData
    };

    fs.appendFileSync(REQUEST_LOG_FILE, JSON.stringify(logEntry) + '\n');
}

function logError(accountNumber, bankCode, error) {
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp,
        accountNumber,
        bankCode,
        error: error.message || String(error)
    };

    fs.appendFileSync(ERROR_LOG_FILE, JSON.stringify(logEntry) + '\n');
}

// Function to hash sensitive data
function hashSensitiveData(data) {
    if (!data) return data;
    return crypto.createHash('sha256').update(data).digest('hex');
}

// Function to validate a single account
async function validateAccount(accountNumber, bankCode) {
    try {
        // For demonstration purposes, let's simulate different API responses
        // This is useful for testing without hitting the actual API constantly

        // Simple deterministic mock based on account number and bank code
        const mockResponse = getMockResponse(accountNumber, bankCode);
        const status = mockResponse.status;
        let data = mockResponse.data;
        let errorMessage = '';

        // Log the mock response
        console.log(`DEBUG: Mock response for ${accountNumber}/${bankCode}: HTTP ${status}, API status: ${data.status || 'Unknown'}`);

        // Handle specific status codes
        if (status === 400) {
            errorMessage = 'Bad request, missing or invalid parameters';
        } else if (status === 404) {
            errorMessage = 'Account or bank not found';
        } else if (status !== 200) {
            errorMessage = `Unexpected status code: ${status}`;
        }

        // Check if account is valid using the response status field
        // This checks the "status" field in the API response
        let isAccountValid = false;
        let apiStatus = 'Unknown';
        let isValidStructure = false;

        if (status === 200 && data) {
            // Mark account as structurally valid (exists in bank)
            isValidStructure = true;

            // Get the API status from the response
            apiStatus = data.status || 'Unknown';

            // Only consider account functionally valid if status field is "Valid"
            isAccountValid = apiStatus === "Valid";

            // Create appropriate messages for different account statuses
            if (apiStatus === "Dormant") {
                errorMessage = "Account is dormant (inactive)";
            } else if (apiStatus === "Post no Credit") {
                errorMessage = "Account cannot receive credits";
            } else if (apiStatus === "Invalid") {
                errorMessage = "Account is invalid in bank system";
            } else if (!isAccountValid && apiStatus) {
                errorMessage = `Account status: ${apiStatus}`;
            }
        }

        // Create comprehensive result object with all API response fields
        return {
            accountNumber,
            bankCode,
            httpStatus: status,
            apiStatus: apiStatus,
            accountHolderName: data.accountHolderName || null,
            bankName: data.bankName || null,
            currency: data.currency || null,
            success: isAccountValid, // Functional validity (status="Valid")
            isValidStructure: isValidStructure, // Structural validity (HTTP 200)
            errorMessage: errorMessage,
            data
        };
    } catch (error) {
        // Log the error
        logError(accountNumber, bankCode, error);

        return {
            accountNumber,
            bankCode,
            httpStatus: 0,
            apiStatus: 'Error',
            accountHolderName: null,
            bankName: null,
            currency: null,
            success: false,
            isValidStructure: false,
            errorMessage: error.message,
            error: error.message
        };
    }
}

// Helper function to generate mock responses
function getMockResponse(accountNumber, bankCode) {
    // Deterministic seed based on account number and bank code
    const seed = (parseInt(accountNumber.replace(/\D/g, '')) || 0) + (parseInt(bankCode) || 0);

    // Use the seed to determine response type
    const responseType = seed % 6; // 0-5 different types

    // Bank names mapping
    const bankNames = {
        '0011': 'Co-operative Bank',
        '0023': 'Equity Bank',
        '0063': 'DTB',
        '0012': 'National Bank',
        '0072': 'UBA Bank',
        '0031': 'Stanbic Bank',
        '0087': 'NCBA Bank',
        '0041': 'KCB Bank',
        '0054': 'Gt Bank',
        '0076': 'Family Bank'
    };

    // Bank name based on bank code or a default
    const bankName = bankNames[bankCode] || 'Unknown Bank';

    // Generate names algorithmically
    const firstNames = ['John', 'Jane', 'Robert', 'Sarah', 'Michael', 'Emily', 'David', 'Emma', 'James', 'Olivia'];
    const lastNames = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor'];
    const nameIndex = seed % 10;
    const accountHolderName = `${firstNames[nameIndex]} ${lastNames[nameIndex]}`;

    // Generate a currency based on the seed
    const currencies = ['KES', 'USD', 'EUR'];
    const currency = currencies[seed % 3];

    // Check if account number has non-numeric characters or wrong length
    const hasNonNumeric = /\D/.test(accountNumber);
    const hasWrongLength = accountNumber.length < 8 || accountNumber.length > 12;
    const invalidBankCode = !bankNames[bankCode];

    if (hasNonNumeric || hasWrongLength || invalidBankCode) {
        // Return 400 Bad Request for structurally invalid accounts
        return {
            status: 400,
            data: { error: 'Invalid account format or bank code' }
        };
    }

    // For structurally valid accounts, return different status types
    if (responseType === 0) {
        // Valid account
        return {
            status: 200,
            data: {
                status: 'Valid',
                accountHolderName: accountHolderName,
                bankName: bankName,
                currency: currency
            }
        };
    } else if (responseType === 1) {
        // Dormant account
        return {
            status: 200,
            data: {
                status: 'Dormant',
                accountHolderName: accountHolderName,
                bankName: bankName,
                currency: currency
            }
        };
    } else if (responseType === 2) {
        // Post no Credit account
        return {
            status: 200,
            data: {
                status: 'Post no Credit',
                accountHolderName: accountHolderName,
                bankName: bankName,
                currency: currency
            }
        };
    } else if (responseType === 3) {
        // Invalid account but HTTP 200
        return {
            status: 200,
            data: {
                status: 'Invalid',
                accountHolderName: accountHolderName,
                bankName: bankName,
                currency: currency
            }
        };
    } else if (responseType === 4) {
        // Account not found
        return {
            status: 404,
            data: { error: 'Account not found' }
        };
    } else {
        // Server error
        return {
            status: 500,
            data: { error: 'Internal server error' }
        };
    }
}

// Add delay between API calls
async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Process accounts in batches
async function processAccounts(accounts, batchSize = 10, delayBetweenBatchesMs = 1000, delayBetweenRequestsMs = 100) {
    const results = [];
    let successCount = 0;
    let validStructureCount = 0;
    let failureCount = 0;

    // Create progress tracker
    const totalBatches = Math.ceil(accounts.length / batchSize);

    console.log(`Starting validation of ${accounts.length} accounts in ${totalBatches} batches...`);
    console.log(`Batch size: ${batchSize}, Delay between batches: ${delayBetweenBatchesMs}ms, Delay between requests: ${delayBetweenRequestsMs}ms`);

    // If we have ML predictions, prioritize accounts by confidence level (validate low confidence first)
    if (accounts.some(acc => acc.hasOwnProperty('ml_prediction'))) {
        console.log('ML predictions found, prioritizing validation based on prediction confidence');

        // Sort by confidence (ascending) so we validate the least confident predictions first
        accounts.sort((a, b) => {
            // If no confidence value or equal confidence, keep original order
            if (!a.hasOwnProperty('ml_confidence') || !b.hasOwnProperty('ml_confidence')) return 0;
            return (a.ml_confidence || 0) - (b.ml_confidence || 0);
        });

        console.log(`Prioritized ${accounts.length} accounts by ML prediction confidence`);
    }

    const startTime = Date.now();

    for (let i = 0; i < accounts.length; i += batchSize) {
        const batch = accounts.slice(i, i + batchSize);
        const batchNumber = Math.floor(i / batchSize) + 1;
        const batchStartItem = i + 1;
        const batchEndItem = Math.min(i + batchSize, accounts.length);

        console.log(`\nProcessing batch ${batchNumber}/${totalBatches} (accounts ${batchStartItem}-${batchEndItem} of ${accounts.length})`);

        // Process each account in the batch sequentially
        for (let j = 0; j < batch.length; j++) {
            const account = batch[j];
            const accountNumber = account['Account Number'];
            const bankCode = account['Bank Code'];
            const overallIndex = i + j + 1;
            const mlPrediction = account.hasOwnProperty('ml_prediction') ?
                `ML prediction: ${account.ml_prediction ? 'VALID' : 'INVALID'} (${account.ml_confidence}% confident)` : '';

            process.stdout.write(`Validating account ${overallIndex}/${accounts.length}: ${accountNumber} (Bank: ${bankCode}) ${mlPrediction}... `);

            const result = await validateAccount(accountNumber, bankCode);

            // Add ML prediction info to the result for comparison
            if (account.hasOwnProperty('ml_prediction')) {
                result.ml_prediction = account.ml_prediction;
                result.ml_confidence = account.ml_confidence;

                // Add information about whether ML prediction matched API result
                result.ml_prediction_correct = (result.success === account.ml_prediction);
            }

            results.push(result);

            if (result.success) {
                successCount++;
                process.stdout.write('✓\n');
            } else {
                failureCount++;
                process.stdout.write('✗\n');
            }

            // Count structurally valid accounts (HTTP 200)
            if (result.isValidStructure) {
                validStructureCount++;
            }

            // Add a small delay between requests
            if (j < batch.length - 1) {
                await sleep(delayBetweenRequestsMs);
            }
        }

        // Print batch summary
        console.log(`Batch ${batchNumber}/${totalBatches} completed. Success: ${successCount}/${results.length} (${(successCount / results.length * 100).toFixed(2)}%)`);

        // Add delay between batches
        if (i + batchSize < accounts.length) {
            const timeElapsed = Math.floor((Date.now() - startTime) / 1000);
            const estimatedTimeRemaining = Math.floor((timeElapsed / results.length) * (accounts.length - results.length));

            console.log(`Progress: ${results.length}/${accounts.length} accounts processed (${(results.length / accounts.length * 100).toFixed(2)}%)`);
            console.log(`Time elapsed: ${timeElapsed}s, Estimated time remaining: ${estimatedTimeRemaining}s`);
            console.log(`Waiting ${delayBetweenBatchesMs}ms before next batch...`);

            await sleep(delayBetweenBatchesMs);
        }
    }

    const totalTime = Math.floor((Date.now() - startTime) / 1000);
    console.log(`\nValidation completed in ${totalTime} seconds`);

    return {
        results,
        summary: {
            total: accounts.length,
            successful: successCount,
            validStructure: validStructureCount,
            failed: failureCount,
            successRate: (successCount / accounts.length * 100).toFixed(2) + '%',
            validStructureRate: (validStructureCount / accounts.length * 100).toFixed(2) + '%',
            processingTimeSeconds: totalTime
        }
    };
}

// Main function
async function main() {
    try {
        let accounts = [];

        // Check if there's a prediction file to use
        if (predictionFile && fs.existsSync(predictionFile)) {
            console.log(`Using ML prediction file: ${predictionFile}`);
            const predictionData = fs.readFileSync(predictionFile, 'utf8');
            accounts = JSON.parse(predictionData);
            console.log(`Loaded ${accounts.length} accounts with ML predictions`);
        } else {
            // Read the CSV file
            console.log('Reading CSV file...');
            const csvData = fs.readFileSync('sample_1000_accounts.csv', 'utf8');
            accounts = parse(csvData, { columns: true, skip_empty_lines: true });
            console.log(`Loaded ${accounts.length} accounts from CSV`);
        }

        // Process the accounts
        const validationResult = await processAccounts(accounts);

        // Calculate summary statistics with detailed status code breakdown
        const httpStatusCounts = {};
        const apiStatusCounts = {};

        // Store accounts by status
        const accountsByStatus = {
            valid: [],
            dormant: [],
            postNoCredit: [],
            invalid: [],
            error: []
        };

        validationResult.results.forEach(result => {
            // Count HTTP status codes
            const httpStatus = result.httpStatus || 'unknown';
            if (!httpStatusCounts[httpStatus]) {
                httpStatusCounts[httpStatus] = 0;
            }
            httpStatusCounts[httpStatus]++;

            // Count API status values
            const apiStatus = result.apiStatus || 'Unknown';
            if (!apiStatusCounts[apiStatus]) {
                apiStatusCounts[apiStatus] = 0;
            }
            apiStatusCounts[apiStatus]++;

            // Group accounts by status
            if (result.success) {
                accountsByStatus.valid.push(result);
            } else if (result.apiStatus === 'Dormant') {
                accountsByStatus.dormant.push(result);
            } else if (result.apiStatus === 'Post no Credit') {
                accountsByStatus.postNoCredit.push(result);
            } else if (result.apiStatus === 'Invalid') {
                accountsByStatus.invalid.push(result);
            } else {
                accountsByStatus.error.push(result);
            }
        });

        // Update the summary with status code details
        validationResult.summary.httpStatusCounts = httpStatusCounts;
        validationResult.summary.apiStatusCounts = apiStatusCounts;

        // Calculate detailed counts for different account types
        const successCount = validationResult.results.filter(r => r.success).length;
        const validStructureCount = validationResult.results.filter(r => r.isValidStructure).length;
        const dormantCount = validationResult.results.filter(r => r.apiStatus === 'Dormant').length;
        const postNoCreditCount = validationResult.results.filter(r => r.apiStatus === 'Post no Credit').length;
        const invalidCount = validationResult.results.filter(r => r.apiStatus === 'Invalid').length;
        const errorCount = validationResult.results.filter(r => !r.isValidStructure).length;

        // Update summary with detailed breakdown
        validationResult.summary.validCount = successCount;
        validationResult.summary.validStructureCount = validStructureCount;
        validationResult.summary.dormantCount = dormantCount;
        validationResult.summary.postNoCreditCount = postNoCreditCount;
        validationResult.summary.invalidCount = invalidCount;
        validationResult.summary.errorCount = errorCount;

        // Calculate rates
        validationResult.summary.validRate = (successCount / validationResult.results.length * 100).toFixed(2) + '%';
        validationResult.summary.validStructureRate = (validStructureCount / validationResult.results.length * 100).toFixed(2) + '%';
        validationResult.summary.dormantRate = (dormantCount / validationResult.results.length * 100).toFixed(2) + '%';
        validationResult.summary.postNoCreditRate = (postNoCreditCount / validationResult.results.length * 100).toFixed(2) + '%';
        validationResult.summary.invalidRate = (invalidCount / validationResult.results.length * 100).toFixed(2) + '%';
        validationResult.summary.errorRate = (errorCount / validationResult.results.length * 100).toFixed(2) + '%';

        // Calculate ML prediction accuracy if available
        if (validationResult.results.some(result => result.hasOwnProperty('ml_prediction'))) {
            const totalPredictions = validationResult.results.filter(r => r.hasOwnProperty('ml_prediction')).length;
            const correctPredictions = validationResult.results.filter(r => r.ml_prediction_correct === true).length;
            const mlAccuracy = (correctPredictions / totalPredictions * 100).toFixed(2);

            validationResult.summary.ml_prediction_accuracy = mlAccuracy + '%';
            validationResult.summary.ml_predictions_correct = correctPredictions;
            validationResult.summary.ml_predictions_total = totalPredictions;

            console.log(`\nML Model Performance:`);
            console.log(`Accuracy: ${mlAccuracy}% (${correctPredictions}/${totalPredictions} correct predictions)`);
        }

        // Create timestamp for output files
        const timestamp = new Date().toISOString().replace(/:/g, '-');

        // Write results to JSON file
        const resultsFileName = `validation_results_${timestamp}.json`;
        fs.writeFileSync(resultsFileName, JSON.stringify(validationResult.results, null, 2));
        console.log(`\nDetailed results written to ${resultsFileName}`);

        // Write summary to JSON file
        const summaryFileName = `validation_summary_${timestamp}.json`;
        fs.writeFileSync(summaryFileName, JSON.stringify(validationResult.summary, null, 2));
        console.log(`Summary written to ${summaryFileName}`);

        // Save accounts by status to separate files for analysis
        if (accountsByStatus.dormant.length > 0) {
            const dormantAccountsFile = `dormant_accounts_${timestamp}.json`;
            fs.writeFileSync(dormantAccountsFile, JSON.stringify(accountsByStatus.dormant, null, 2));
            console.log(`Dormant accounts written to ${dormantAccountsFile}`);
        }

        if (accountsByStatus.postNoCredit.length > 0) {
            const postNoCreditAccountsFile = `post_no_credit_accounts_${timestamp}.json`;
            fs.writeFileSync(postNoCreditAccountsFile, JSON.stringify(accountsByStatus.postNoCredit, null, 2));
            console.log(`Post No Credit accounts written to ${postNoCreditAccountsFile}`);
        }

        if (accountsByStatus.invalid.length > 0) {
            const invalidAccountsFile = `invalid_accounts_${timestamp}.json`;
            fs.writeFileSync(invalidAccountsFile, JSON.stringify(accountsByStatus.invalid, null, 2));
            console.log(`Invalid accounts written to ${invalidAccountsFile}`);
        }

        if (accountsByStatus.error.length > 0) {
            const errorAccountsFile = `error_accounts_${timestamp}.json`;
            fs.writeFileSync(errorAccountsFile, JSON.stringify(accountsByStatus.error, null, 2));
            console.log(`Error accounts written to ${errorAccountsFile}`);
        }

        console.log('\nValidation Summary:');
        console.log(`Total accounts: ${validationResult.summary.total}`);
        console.log(`Structurally valid accounts (HTTP 200): ${validStructureCount} (${validationResult.summary.validStructureRate})`);
        console.log(`Functionally valid accounts (status="Valid"): ${successCount} (${validationResult.summary.validRate})`);
        console.log(`Dormant accounts: ${dormantCount} (${validationResult.summary.dormantRate})`);
        console.log(`Post No Credit accounts: ${postNoCreditCount} (${validationResult.summary.postNoCreditRate})`);
        console.log(`Invalid accounts (in bank system): ${invalidCount} (${validationResult.summary.invalidRate})`);
        console.log(`Error accounts (HTTP errors): ${errorCount} (${validationResult.summary.errorRate})`);

        // Show detailed HTTP status code breakdown
        console.log('\nHTTP Status Code Breakdown:');
        for (const [code, count] of Object.entries(httpStatusCounts)) {
            console.log(`  HTTP ${code}: ${count} accounts (${(count / validationResult.summary.total * 100).toFixed(2)}%)`);
        }

        // Show detailed API status breakdown
        console.log('\nAPI Status Breakdown:');
        for (const [status, count] of Object.entries(apiStatusCounts)) {
            console.log(`  ${status}: ${count} accounts (${(count / validationResult.summary.total * 100).toFixed(2)}%)`);
        }

        console.log(`\nProcessing time: ${validationResult.summary.processingTimeSeconds} seconds`);

        if (validationResult.summary.hasOwnProperty('ml_prediction_accuracy')) {
            console.log(`ML prediction accuracy: ${validationResult.summary.ml_prediction_accuracy}`);
        }
    } catch (error) {
        console.error(`Error in main process: ${error.message}`);
        console.error(error.stack);
    }
}

main(); 