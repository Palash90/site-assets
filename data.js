const cdn = "https://palash90.github.io/site-assets/"
const path_var = "/new_site/#"
const data = {
    name: "Palash Kanti Kundu",
    shortName: "Palash",
    labels: {
        contents: "Blogs",
        projects: "Projects",
        swe: "Software Engineering",
        music: "Music",
        contentNotExists: "Content does not exist"
    },
    pages: {
        contents: {
            intro: "Software Engineering is what I do for a living, music is what I do to live. Here you can find both."
        },
        home: {
            tag: "Crafting code, creating harmony.",
            greeting: "Hi, I'm ",
            moto: "My moto: ",
            motos: ["Code", "Create", "Inspire"],
            desc: "I’m a software engineer with 13 years of experience, specializing in tech stacks like Java, Python, C#, React.js and Rust. Apart from my day-to-day office work, I’ve built a machine learning library, implemented parts of a distributed system, few react.js based games, and am passionate about system design. I also contribute to community through my blogs to share insights and learnings. \n \n  Outside of software engineering, I pursue music, specifically creating guitalele tutorials, which helps me stay creative and balanced.",
            profilePicUrl:""
        },
        projects: {
            intro: "If any idea catches my attention, I try to implement the same.",
            techStack: "Technology: "
        },
        about: {
            mdUrl: cdn + "ME.md"
        }
    },
    navLinks: [
        { label: "Home", link: path_var + "/" },
        { label: "Blogs", link: path_var + "/contents" },
        { label: "Projects", link: path_var + "/projects" },
        { label: "About", link: path_var + "/about" }
    ],
    contents: contents,
    projects: projects,
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

