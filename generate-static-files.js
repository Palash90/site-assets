const fs = require('fs');

function mergeCode(files, fileName) {
    var mergedCode = files
        .map(path => fs.readFileSync(path, 'utf-8'))
        .join('\n');

    fs.writeFileSync(fileName, mergedCode);
}

var jsFiles = [
    "./site-contents/projects.js",
    "./site-contents/contents.js",
    "./site-contents/data.js",
    "./site-contents/findProps.js"
];

var cssFiles = [
    "./site-contents/app.css",
    "./site-contents/about.css",
    "./site-contents/tech-blog.css",
    "./site-contents/tech-blog-2.css",
    "./site-contents/music-blog.css"
];

mergeCode(jsFiles, "script.js")
mergeCode(cssFiles, "styles.css")
