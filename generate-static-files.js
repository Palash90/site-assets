const fs = require('fs');

function mergeCode(files, fileName) {
    var mergedCode = files
        .map(path => fs.readFileSync(path, 'utf-8'))
        .join('\n');

    fs.writeFileSync(fileName, mergedCode);
}

mergeCode(["./site-contents/projects.js", "./site-contents/contents.js", "./site-contents/data.js", "./site-contents/findProps.js"], "script.js")
mergeCode(["./site-contents/about.css", "./site-contents/blog.css"], "styles.css")
