const fs = require('fs');

function mergeCode(files, fileName) {
    var mergedCode = files
        .map(path => fs.readFileSync(path, 'utf-8'))
        .join('\n');

    fs.writeFileSync(fileName, mergedCode);
}

var jsFiles = [
    "./site-contents/js/common.js",
    "./site-contents/js/contentArrayManipulator.js",
    "./site-contents/js/techBlogs.js",
    "./site-contents/js/musicBlogs.js",
    "./site-contents/js/contents.js",
    "./site-contents/js/projects.js",
    "./site-contents/js/data.js",
    "./site-contents/js/findProps.js",
    "./site-contents/js/generateConsolidatedScript.js"
];

var cssFiles = [
    "./site-contents/stylesheets/app.css",
    "./site-contents/stylesheets/markdown.css",
    "./site-contents/stylesheets/markdown-tomato.css",
    "./site-contents/stylesheets/markdown-cyan.css",
    "./site-contents/stylesheets/about.css",
    "./site-contents/stylesheets/tech-blog.css",
    "./site-contents/stylesheets/music-blog-2.css",
    "./site-contents/stylesheets/about-2.css"
];

mergeCode(jsFiles, "consolidated_script.js")
mergeCode(cssFiles, "styles.css")
