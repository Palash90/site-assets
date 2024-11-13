const fs = require('fs');
const path = require('path');

// Map of variables and their replacement values
const variableMap = {
    'setting-up-a-site-part-1-home-page': 'https://palash90.github.io/site-assets/blogs/setting-up-a-site/part-1.md',
    'some_other_variable': 'value_for_some_other_variable',
};

// Function to replace patterns in a file
const replaceVariablesInFile = (filePath) => {
    // Read the file content
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            console.error(`Error reading file ${filePath}: ${err}`);
            return;
        }

        // Regex to match ${some_variable} or ${some_other_variable}
        let modifiedData = data;
        Object.keys(variableMap).forEach((key) => {
            const regex = new RegExp(`\\$\\{${key}\\}`, 'g'); // Match ${some_variable}
            modifiedData = modifiedData.replace(regex, variableMap[key]);
        });

        // Write the modified data back to the file
        fs.writeFile(filePath, modifiedData, 'utf8', (err) => {
            if (err) {
                console.error(`Error writing file ${filePath}: ${err}`);
            } else {
                console.log(`File updated: ${filePath}`);
            }
        });
    });
};

// Function to recursively read the directory
const processDirectory = (dirPath) => {
    fs.readdir(dirPath, { withFileTypes: true }, (err, files) => {
        if (err) {
            console.error(`Error reading directory ${dirPath}: ${err}`);
            return;
        }

        files.forEach((file) => {
            const fullPath = path.join(dirPath, file.name);

            if (file.isDirectory()) {
                // Recursively process directories
                processDirectory(fullPath);
            } else {
                // Process file
                if (path.extname(file.name) === ".md") {
                    replaceVariablesInFile(fullPath);
                }
            }
        });
    });
};

// Example: Start processing the directory
const directoryPath = './blogs';
processDirectory(directoryPath);
