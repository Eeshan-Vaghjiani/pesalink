const fs = require('fs');
const https = require('https');

// API Configuration
const API_KEY = '609dab841280674f1a780272f59e9e4e';
const API_URL = 'account-validation-service.dev.pesalink.co.ke';

// Function to validate an account
function validateAccount(accountNumber, bankCode) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify({
            accountNumber: accountNumber,
            bankCode: bankCode
        });

        const options = {
            hostname: API_URL,
            port: 443,
            path: '/api/validate',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': API_KEY,
                'Content-Length': data.length
            }
        };

        console.log('Request options:', JSON.stringify(options, null, 2));

        const req = https.request(options, (res) => {
            let responseData = '';

            res.on('data', (chunk) => {
                responseData += chunk;
            });

            res.on('end', () => {
                console.log('Raw response:', responseData);
                try {
                    const result = JSON.parse(responseData);
                    resolve({
                        statusCode: res.statusCode,
                        body: result
                    });
                } catch (e) {
                    resolve({
                        statusCode: res.statusCode,
                        body: responseData
                    });
                }
            });
        });

        req.on('error', (error) => {
            reject(error);
        });

        req.write(data);
        req.end();
    });
}

// Read the CSV file
fs.readFile('sample_1000_accounts.csv', 'utf8', async (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }

    // Parse CSV content
    const lines = data.trim().split('\n');
    const headers = lines[0].split(',');

    // Test with first 2 accounts
    const testAccounts = [];
    for (let i = 1; i <= 2; i++) {
        if (i < lines.length) {
            const values = lines[i].split(',');
            testAccounts.push({
                accountNumber: values[0],
                bankCode: values[1]
            });
        }
    }

    console.log('Testing validation with accounts:');
    console.log(testAccounts);

    // Test each account
    for (const account of testAccounts) {
        try {
            console.log(`\nValidating account: ${account.accountNumber}, bank code: ${account.bankCode}`);
            const result = await validateAccount(account.accountNumber, account.bankCode);
            console.log(`Status code: ${result.statusCode}`);
            console.log('Response:', JSON.stringify(result.body, null, 2));
        } catch (error) {
            console.error('Error validating account:', error);
        }
    }
}); 