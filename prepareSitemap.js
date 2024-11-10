const siteUrl = "https://palashkantikundu.in/"

const staticLinks = [
    "https://palashkantikundu.in/",
    "https://www.palashkantikundu.in/",
    "https://tech.palashkantikundu.in/",
    "https://vi-essentials.palashkantikundu.in/",
    "https://go-essentials.palashkantikundu.in/"
]

const allNavLinks = findProp("navLinks").map(n => siteUrl + n.link)

const allBlogs = findProp("contents.swe").concat(findProp("contents.music")).map(
    b => siteUrl + "#/content/" + b.id
)

const allProjects = findProp("projects").filter(p => p.type === "react" && p.playUrl).map(p => siteUrl + p.playUrl)


const allLinks = [...staticLinks, ...allNavLinks, ...allBlogs, ...allProjects]

console.log(allLinks)
