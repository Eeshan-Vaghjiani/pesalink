const fs = require('fs');
const fetch = require('node-fetch');

// API Configuration
const API_KEY = '609dab841280674f1a780272f59e9e4e';
const API_URL = 'https://account-validation-service.dev.pesalink.co.ke/api/validate';

// Function to validate an account with different auth methods
async function validateWithDifferentAuthMethods(accountNumber, bankCode) {
    const authMethods = [
        { name: "Plain API Key", headers: { 'Authorization': API_KEY } },
        { name: "Bearer Token", headers: { 'Authorization': `Bearer ${API_KEY}` } },
        { name: "API-Key Header", headers: { 'API-Key': API_KEY } },
        { name: "X-API-Key Header", headers: { 'X-API-Key': API_KEY } },
        { name: "Query Parameter", url: `${API_URL}?apiKey=${API_KEY}`, headers: {} }
    ];

    for (const method of authMethods) {
        console.log(`\nTrying auth method: ${method.name}`);
        try {
            const url = method.url || API_URL;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    ...method.headers
                },
                body: JSON.stringify({
                    accountNumber,
                    bankCode
                })
            });

            const status = response.status;
            const data = await response.text();

            console.log('Status:', status);
            console.log('Raw response:', data);

            if (status === 200) {
                console.log('Success! Working auth method found:', method.name);
                return {
                    success: true,
                    authMethod: method.name,
                    statusCode: status,
                    body: JSON.parse(data)
                };
            }
        } catch (error) {
            console.error(`Error with auth method ${method.name}:`, error.message);
        }
    }

    return { success: false, message: "All auth methods failed" };
}

// Main function
async function main() {
    try {
        // Read the CSV file
        const data = fs.readFileSync('sample_1000_accounts.csv', 'utf8');

        // Parse CSV content
        const lines = data.trim().split('\n');

        // Test with first account
        const values = lines[1].split(',');
        const testAccount = {
            accountNumber: values[0],
            bankCode: values[1]
        };

        console.log('Testing validation with account:', testAccount);

        // Try different auth methods
        const result = await validateWithDifferentAuthMethods(
            testAccount.accountNumber,
            testAccount.bankCode
        );

        console.log('\nFinal result:', JSON.stringify(result, null, 2));
    } catch (error) {
        console.error('Error in main function:', error);
    }
}

main(); 