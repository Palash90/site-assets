const fs = require('fs');

const json = JSON.stringify(JSON.stringify(data))

var findPropsStr = fs.readFileSync("./site-contents/js/findProps.js");

var code = "const data = JSON.parse(" + json + ")" + "\n"

code += findPropsStr + "\n"
code +="console.log(findProp('pages.home.desc'))"


console.log("Writing consolidated code to file")
fs.writeFileSync("script.js", code)

