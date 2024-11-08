const data = {
    name: "Palash Kanti Kundu",
    shortName: "Palash",
    labels: {
        blogs: "Contents",
        projects: "Projects",
        swe: "Software Engineering",
        music: "Music",
        blogNotExists: "Blog does not exist"
    },
    contents: {
        blogs: {
            intro: "Software Engineering is what I do for a living, music is what I do to live. Here you can find both."
        },
        home: {
            tag: "Crafting code, creating harmony.",
            greeting: "Hi, I'm ",
            moto: "My moto: ",
            motos: ["Code", "Create", "Inspire"],
            desc: "I’m a software engineer with 13 years of experience, specializing in tech stacks like Java, Python, C#, React.js and Rust. Apart from my day-to-day office work, I’ve built a machine learning library, implemented parts of a distributed system, few react.js based games, and am passionate about system design. I also contribute to community through my blogs to share insights and learnings. \n \n  Outside of software engineering, I pursue music, specifically creating guitalele tutorials, which helps me stay creative and balanced."
        }
    },
    navLinks: [
        { label: "Home", link: "/" },
        { label: "Contents", link: "/contents" },
        { label: "Projects", link: "/projects" },
        { label: "About", link: "/about" }
    ],
    blogs: {
        swe: [
            {
                id: "static-file-hosting",
                title: "Simple Static File Hosting",
                url: "https://palash90.github.io/site-assets/blogs/static-file-hosting/README.md",
                publishDate: "November 08, 2024",
                type: "markdown"
            }
        ],
        music: [
            { id: 1, publishDate: "Jan 2014", title: "Youtube", url: "hMBpAPGqX8k", type: "video" },
            { id: 2, publishDate: "Jan 2014", title: "Both", video: "hMBpAPGqX8k", md: "https://palash90.github.io/site-assets/blogs/static-file-hosting/README.md", type: "both" }
        ]
    },
    projects: [
        { id: 1, name: "Iron Learn", desc: "A pure Rust Machine Learning Library", type: "Rust", url: "https://github.com/Palash90/iron_learn" },
        { id: 1, name: "HDL Emulator", desc: "An online Web Based HDL Emulator for Teaching Purpose", type: "Component", url: "/component/hdlEmulator" }
    ],
    about: {
        jobs: [
            { name: "Cognizant Technology Solutions", start: "December 2022", end: "Present", location: "Bengaluru, India", role: "Cloud Deverloper III" },
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

function findProp(prop, defval) {
    var obj = data
    prop = prop.split('.');
    for (var i = 0; i < prop.length; i++) {
        if (typeof obj[prop[i]] == 'undefined')
            return null;
        obj = obj[prop[i]];
    }
    return obj;
}