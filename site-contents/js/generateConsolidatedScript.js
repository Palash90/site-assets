const fs = require('fs');

// Write data.json — only contents key; common config + findProp are in the Firebase app
fs.writeFileSync("data.json", JSON.stringify({ contents: data.contents }, null, 2))
console.log("Writing data.json")