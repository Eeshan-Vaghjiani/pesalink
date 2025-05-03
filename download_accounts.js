const fs = require('fs');
const fetch = require('node-fetch');
const path = require('path');

// API Configuration
const API_KEY = '609dab841280674f1a780272f59e9e4e';
const BASE_URL = 'https://account-validation-service.dev.pesalink.co.ke';
const DOWNLOAD_ENDPOINT = '/download/sample_1000';
const OUTPUT_FILE = 'sample_1000_accounts.csv';

async function downloadCsvFile() {
    console.log(`Downloading latest accounts data from API...`);

    try {
        const response = await fetch(`${BASE_URL}${DOWNLOAD_ENDPOINT}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/csv',
                'Authorization': API_KEY
            }
        });

        if (!response.ok) {
            throw new Error(`API responded with status: ${response.status} - ${response.statusText}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('text/csv')) {
            throw new Error(`Expected CSV but got content type: ${contentType}`);
        }

        // Get file content as text
        const csvContent = await response.text();

        // Save to file
        fs.writeFileSync(OUTPUT_FILE, csvContent);

        console.log(`✅ Successfully downloaded latest account data to ${OUTPUT_FILE}`);
        console.log(`File size: ${(csvContent.length / 1024).toFixed(2)} KB`);

        // Display sample of the data
        const lines = csvContent.split('\n');
        console.log(`\nTotal records: ${lines.length - 1}`);
        console.log(`\nSample data (first 5 records):`);
        console.log(lines.slice(0, 6).join('\n'));

        return true;
    } catch (error) {
        console.error(`❌ Error downloading CSV file: ${error.message}`);

        // Try alternate approach with authorization as Bearer token
        try {
            console.log('Trying alternate authorization method...');
            const retryResponse = await fetch(`${BASE_URL}${DOWNLOAD_ENDPOINT}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/csv',
                    'Authorization': `Bearer ${API_KEY}`
                }
            });

            if (!retryResponse.ok) {
                throw new Error(`API responded with status: ${retryResponse.status} - ${retryResponse.statusText}`);
            }

            const csvContent = await retryResponse.text();
            fs.writeFileSync(OUTPUT_FILE, csvContent);

            console.log(`✅ Successfully downloaded latest account data using Bearer token to ${OUTPUT_FILE}`);
            console.log(`File size: ${(csvContent.length / 1024).toFixed(2)} KB`);

            // Display sample of the data
            const lines = csvContent.split('\n');
            console.log(`\nTotal records: ${lines.length - 1}`);
            console.log(`\nSample data (first 5 records):`);
            console.log(lines.slice(0, 6).join('\n'));

            return true;
        } catch (retryError) {
            console.error(`❌ Alternate approach also failed: ${retryError.message}`);

            if (fs.existsSync(OUTPUT_FILE)) {
                console.log(`ℹ️ Using existing ${OUTPUT_FILE} file.`);
                return false;
            } else {
                console.error(`❌ No existing CSV file found. Account validation will not work correctly.`);
                return false;
            }
        }
    }
}

// Execute the download
downloadCsvFile().then(success => {
    if (success) {
        console.log('CSV file download completed successfully.');
    } else {
        console.log('CSV file download failed. Check errors above.');
    }
}); 