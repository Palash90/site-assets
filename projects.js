const raw_content_cdn = "https://raw.githubusercontent.com/Palash90";
const path_prefix = "/#"

const projects = [
    {
        id: "iron-learn",
        name: "Iron Learn",
        desc: "A pure Rust Machine Learning Library",
        type: "rust",
        githubUrl: "https://github.com/Palash90/iron_learn",
        mdUrl: raw_content_cdn + "/iron_learn/refs/heads/main/README.md"
    },
    {
        id: "hdl-emulator",
        name: "HDL Emulator",
        desc: "An online Web Based HDL Emulator for Teaching Purpose",
        type: "javascript",
        playUrl: "https://emulator.palashkantikundu.in/",
        githubUrl: "https://github.com/Palash90/emulator",
        mdUrl: raw_content_cdn + "/emulator/refs/heads/main/README.md"
    },
    {
        id: "dist-fs",
        name: "Distributed File System",
        desc: "A simple distributed file system implementation",
        type: "java",
        githubUrl: "https://github.com/Palash90/dist-fs",
        mdUrl: raw_content_cdn + "/dist-fs/refs/heads/main/README.md"
    },
    {
        id: "tic-tac-slide",
        name: "Tic Tac Slide",
        desc: "An innovation over simple tic tac toe for added fun",
        type: "react",
        playUrl: path_prefix + "/component/tic-tac-slide",
        githubUrl: ""
    }
]