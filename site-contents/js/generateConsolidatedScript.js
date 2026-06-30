const fs = require('fs');

const json = JSON.stringify(JSON.stringify(data))

// Write script.js (backward compat — data + findProp function)
var findPropsStr = fs.readFileSync("./site-contents/js/findProps.js");
var code = "const data = JSON.parse(" + json + ")" + "\n"
code += findPropsStr + "\n"
console.log("Writing consolidated code to file")
fs.writeFileSync("script.js", code)

// Write data.json — only contents key; common config is now in the Firebase app (baseConfig.js)
fs.writeFileSync("data.json", JSON.stringify({ contents: data.contents }, null, 2))
console.log("Writing data.json")