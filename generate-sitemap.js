const fs = require('fs');
const vm = require('vm');

const files = [
    './projects.js',
    './contents.js',
    './data.js',
    './findProps.js',
    './prepareSitemap.js',
];
let sandbox = {
    require,
    console
};

vm.createContext();
const mergedCode = files
    .map(path => fs.readFileSync(path, 'utf-8'))
    .join('\n');
try {
    vm.runInNewContext(mergedCode, sandbox);
} catch (error) {
    console.error('Error executing merged code:', error);
}

console.log("Done");
