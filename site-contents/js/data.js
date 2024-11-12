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
            itemsPerPage: 50,

            techBlogClass: "tech-blog-content markdown markdown-tomato",
            musicBlogClass: "music-blog-2 markdown markdown-tomato",
            genericBlogClass: "blog-content markdown markdown-tomato",

            sweHeadColor: getCommon('themeHeaderColor'),
            musicHeadColor: getCommon('themeHeaderColor'),
            h1Color: getCommon('themeHeaderColor'),
            pColor: getCommon('themePColor'),

            linkClass: "tech-blog-2"
        },
        home: {
            tag: "Crafting code, creating harmony.",
            greeting: "Hi, I'm ",
            moto: "My moto: ",
            motos: ["Code", "Create", "Inspire"],
            desc: "I’m a software engineer with 13 years of experience, specializing in tech stacks like Java, Python, C#, React.js and Rust. Apart from my day-to-day office work, I’ve built a machine learning library, implemented parts of a distributed system, few react.js based games, and am passionate about system design. I also contribute to community through my blogs to share insights and learnings. \n \n  Outside of software engineering, I pursue music, specifically creating guitalele tutorials, which helps me stay creative and balanced.\n\n",
            profilePicUrl: getCommon("cdn") + "/assets/profile.jpg",

            mainStyle: "main-app-lora custom-theme-tomato",
            secondaryStyle: "content-body-opensans",
            class: "",

            techBlogHeader: "Recent Tech Blogs | ",
            techBlogTag: "Things I learned or implemented recently",
            techBlogShowAll: "Show All",

            musicBlogHeader: "Recent Music Blogs | ",
            musicBlogShowAll: "Show All",
            musicBlogTag: "Things I explored recently",

            h1Color: getCommon('themeHeaderColor'),
            pColor: getCommon('themePColor')
        },
        projects: {
            intro: "If any idea catches my attention, I try to implement the same.",
            techStack: "Technology: ",

            h1Color: getCommon('themeHeaderColor'),
            pColor: getCommon('themePColor')
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
        { label: "About", link: getCommon("path_var") + "/about" },
        { label: "Sitemap", link: "https://sitemap.palashkantikundu.in/" }
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

