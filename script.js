const staticMap = {
    cdn: "https://palash90.github.io/site-assets",
    path_var: "/#",
    raw_content_cdn: "https://raw.githubusercontent.com/Palash90"
}

const getCommon = (key) => staticMap[key]
function getLatestDate(date1, date2) {
    // Check if both dates are null or undefined
    if (!date1 && !date2) {
        return null;
    }

    // If only one date is null or undefined, return the other date
    if (!date1) {
        return date2;
    }

    if (!date2) {
        return date1;
    }

    // Compare the dates and return the latest one
    return date1 > date2 ? date1 : date2;
}

const getDateString = (date) => {
    if (!date) {
        return date
    }

    let year = new Intl.DateTimeFormat('en', { year: 'numeric' }).format(date);
    let month = new Intl.DateTimeFormat('en', { month: 'short' }).format(date);
    let day = new Intl.DateTimeFormat('en', { day: '2-digit' }).format(date);

    return `${day}-${month}-${year}`;
}

const modifyArray = (arr) => {

    arr = arr.map(c => {
        c.publishDate = c.publishDate ? new Date(c.publishDate) : undefined
        c.lastUpdated = c.lastUpdated ? new Date(c.lastUpdated) : undefined

        c.sortBy = getLatestDate(c.publishDate, c.lastUpdated)
        return c
    })

    arr.sort((a, b) => b.sortBy - a.sortBy)

    return arr.map(c => {
        return {
            id: c.id,
            title: c.title,
            publishDate: getDateString(c.publishDate),
            lastUpdated: getDateString(c.lastUpdated),
            mdUrl: c.mdUrl,
            videoId: c.videoId
        }
    })
}


var sweContents = [
    {
        id: "static-file-hosting",
        title: "Simple Static File Hosting",
        mdUrl: getCommon("cdn") + "/blogs/static-file-hosting/README.md",
        publishDate: "Nov 08, 2021"
    },
    {
        id: "grafana-on-aws",
        title: "Configure Grafana on AWS",
        mdUrl: getCommon("cdn") + "/blogs/aws-grafana/README.md",
        publishDate: "Jun 11, 2024"
    },
    {
        id: "vi-essentials",
        title: "vi Essentials",
        publishDate: "Nov 01, 2020",
        mdUrl: getCommon("cdn") + "/blogs/vi/README.md"
    },
    {
        id: "go-essentials",
        title: "Go Essentials Both",
        publishDate: "Nov 01, 2020",
        lastUpdated: "Nov 12, 2024",
        mdUrl: getCommon("cdn") + "/blogs/go-tut/README.md",
        videoId: "hMBpAPGqX8k"
    }
];

sweContents = modifyArray(sweContents)


var musicContents = [
    {
        id: "sliding-shape-guitalele",
        publishDate: "Oct 16, 2024",
        title: "Sliding Shape Guitalele",
        videoId: "hMBpAPGqX8k"
    }
]

musicContents = modifyArray(musicContents)

const contents = {
    swe: sweContents,
    music: musicContents
}
const projects = [
    {
        id: "iron-learn",
        name: "Iron Learn",
        desc: "A pure Rust Machine Learning Library",
        type: "rust",
        githubUrl: "https://github.com/Palash90/iron_learn",
        mdUrl: getCommon('raw_content_cdn') + "/iron_learn/refs/heads/main/README.md"
    },
    {
        id: "hdl-emulator",
        name: "HDL Emulator",
        desc: "An online Web Based HDL Emulator for Teaching Purpose",
        type: "javascript",
        playUrl: "https://emulator.palashkantikundu.in/",
        githubUrl: "https://github.com/Palash90/emulator",
        mdUrl: getCommon('raw_content_cdn') + "/emulator/refs/heads/main/README.md"
    },
    {
        id: "dist-fs",
        name: "Distributed File System",
        desc: "A simple distributed file system implementation",
        type: "java",
        githubUrl: "https://github.com/Palash90/dist-fs",
        mdUrl: getCommon('raw_content_cdn') + "/dist-fs/refs/heads/main/README.md"
    },
    {
        id: "tic-tac-slide",
        name: "Tic Tac Slide",
        desc: "An innovation over simple tic tac toe for added fun",
        type: "react",
        playUrl: getCommon('path_var') + "/component/tic-tac-slide",
        githubUrl: ""
    }
]
const data = {
    name: "Palash Kanti Kundu",
    shortName: "Palash",
    labels: {
        contents: "Blogs",
        projects: "Projects",
        swe: "Software Engineering",
        music: "Music",
        contentNotExists: "Content does not exist",
        publishedOn: "Published on: ",
        lastUpdated: "Last updated on: "
    },
    pages: {
        contents: {
            header: "Blogs",
            techHeader: "Tech Blogs",
            musicHeader: "Music Blogs",
            intro: "Software Engineering is what I do for a living, music is what I do to live. Here you can find both.",
            techIntro: "Code, Insights, and the Art of Problem-Solving.",
            musicIntro: "Exploring the World of Music, One Post at a Time.",
            h1Color: "tomato",
            pColor: "",
            sweHeadColor: "#00b0ff",
            musicHeadColor: "#1abc9c",
            itemsPerPage: 50,
            techBlogClass: "tech-blog-content",
            musicBlogClass: "music-blog-2   ",
            genericBlogClass: "blog-content",
            linkClass: "tech-blog-2"
        },
        home: {
            tag: "Crafting code, creating harmony.",
            greeting: "Hi, I'm ",
            moto: "My moto: ",
            motos: ["Code", "Create", "Inspire"],
            desc: "I’m a software engineer with 13 years of experience, specializing in tech stacks like Java, Python, C#, React.js and Rust. Apart from my day-to-day office work, I’ve built a machine learning library, implemented parts of a distributed system, few react.js based games, and am passionate about system design. I also contribute to community through my blogs to share insights and learnings. \n \n  Outside of software engineering, I pursue music, specifically creating guitalele tutorials, which helps me stay creative and balanced.\n\n",
            profilePicUrl: getCommon("cdn") + "/assets/profile.jpg",
            h1Color: "#00b0ff",
            pColor: "#1abc9c",
            mainStyle: "main-app-lora",
            secondaryStyle: "content-body-opensans",
            class: "",

            techBlogHeader: "Recent Tech Blogs | ",
            techBlogTag: "Things I learned or implemented recently",
            techBlogShowAll: "Show All",

            musicBlogHeader: "Recent Music Blogs | ",
            musicBlogShowAll: "Show All",
            musicBlogTag: "Things I explored recently"
        },
        projects: {
            intro: "If any idea catches my attention, I try to implement the same.",
            techStack: "Technology: ",
            h1Color: "#1abc9c",
            pColor: "",
            titleColor: "#3498db",
            bodyColor: ""
        },
        about: {
            mdUrl: getCommon("cdn") + "/assets/ME.md",
            resume: getCommon("cdn") + "Palash_Kanti_Kundu_13YOE_BackEnd_AI.pdf",
            resumeName: "Palash_Kanti_Kundu_13YOE_BackEnd_AI.pdf",
            blogClass: "about-2"
        }
    },
    techStack: {
        iconColor: {
            rust: "#B7410E",
            java: "blue",
            javascript: "red",
            react: "teal"
        },
        title: {
            rust: "Rust",
            java: "Java",
            javascript: "JavaScript",
            react: "React.js"
        }
    },
    navLinks: [
        { label: "Home", link: getCommon("path_var") + "/" },
        { label: "Tech Blogs", link: getCommon("path_var") + "/contents/tech" },
        { label: "Music Blogs", link: getCommon("path_var") + "/contents/music" },
        { label: "Projects", link: getCommon("path_var") + "/projects" },
        { label: "About", link: getCommon("path_var") + "/about" }
    ],
    contents: contents,
    projects: projects,
    about: {
        jobs: [
            { name: "Hewlett Packard Enterprise", start: "December 2022", end: "Present", location: "Bengaluru, India", role: "Cloud Deverloper III" },
            { name: "GE Renewable Energy", start: "December 2018", end: "November 2022", location: "Hyderabad, India", role: "Senior Software Engineer" },
            { name: "Oracle India Pvt. Ltd.", start: "January 2016", end: "December 2018", location: "Hyderabad, India", role: "Senior Applications Engineer" },
            { name: "HCL Technologies", start: "November 2014", end: "November 2015", location: "Kolkata, India", role: "Senior Software Engineer" },
            { name: "Cognizant Technology Solutions", start: "November 2011", end: "October 2014", location: "Kolkata, India", role: "Programmer Analyst" },
        ],
        education: [
            { institution: "College of Engineering and Management, Kolaghat", start: "July 2007", end: "Jun 2011", degree: "Bachelor of Technology" }
        ]
    }
}


const fs = require('fs');

const json = JSON.stringify(JSON.stringify(data))

var findPropsStr = fs.readFileSync("./site-contents/js/findProps.js");

var code = "const data = JSON.parse(" + json + ")" + "\n"

code += findPropsStr + "\n"
code +="console.log(findProp('pages.home.desc'))"


fs.writeFileSync("script_1.js", code)

