const fs = require('fs');

const json = JSON.stringify(JSON.stringify(data))

var findPropsStr = fs.readFileSync("./site-contents/js/findProps.js");

var code = "const data = JSON.parse(" + json + ")" + "\n"

code += findPropsStr + "\n"

console.log("Writing consolidated code to file")
fs.writeFileSync("script.js", code)

process.exit(1)