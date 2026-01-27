//const fs = require('fs');
const siteUrl = "https://palashkantikundu.in/"

const staticLinks = [
    "https://palashkantikundu.in/",
    "https://www.palashkantikundu.in/",
    "https://tech.palashkantikundu.in/",
    "https://music.palashkantikundu.in/",
    "https://guitalele-tutorials.palashkantikundu.in/",
    "https://vi-essentials.palashkantikundu.in/",
    "https://go-essentials.palashkantikundu.in/",
    "https://ai.palashkantikundu.in",
    "https://ai.palashkantikundu.in/visualizers/linear-regression.html",
    "https://ai.palashkantikundu.in/visualizers/neural-network.html",
]

const allNavLinks = findProp("navLinks").map(n => siteUrl + n.link)

const allBlogs = findProp("contents.swe").concat(findProp("contents.music")).map(
    b => siteUrl + "#/content/" + b.id
)

const allProjects = findProp("projects").filter(p => p.type === "react" && p.playUrl).map(p => siteUrl + p.playUrl)


const allLinks = [...staticLinks, ...allNavLinks, ...allBlogs, ...allProjects]

var siteMap = '<?xml version="1.0" encoding="utf-8"?>'
siteMap += '<?xml-stylesheet type="text/xsl" href="https://palash90.github.io/site-assets/xsl/vanilla-water.xslt" ?>'
siteMap += '\n'
siteMap += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
siteMap += '\n'

allLinks.map(l => {
    if(l.indexOf('sitemap') === -1) {
        siteMap += "\t<url>"
        siteMap += '\n'
        siteMap += '\t\t<loc>' + l + '</loc>'
        siteMap += '\n'
        siteMap += '\t</url>'
        siteMap += '\n'
    }
})

siteMap += "</urlset>"

try {
    fs.writeFileSync('sitemap.xml', siteMap);
    console.log('Sitemap file written successfully');
} catch (err) {
    console.error(err);
}
