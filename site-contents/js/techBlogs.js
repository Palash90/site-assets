var sweContents = [
  {
    id: "static-file-hosting",
    title: "Simple Static File Hosting",
    mdUrl: getCommon("cdn") + "/target/blogs/static-file-hosting/README.md",
    publishDate: "Nov 08, 2024",
  },
  {
    id: "grafana-on-aws",
    title: "Configure Grafana on AWS",
    mdUrl: getCommon("cdn") + "/target//blogs/aws-grafana/README.md",
    publishDate: "Jun 11, 2024",
  },
  {
    id: "vi-essentials",
    title: "vi Essentials",
    publishDate: "Nov 01, 2020",
    mdUrl: getCommon("cdn") + "/target/blogs/vi/README.md",
  },
  {
    id: "go-essentials",
    title: "Go Essentials",
    publishDate: "Nov 01, 2020",
    mdUrl: getCommon("cdn") + "/target/blogs/go-tut/README.md",
  },
  {
    id: "setting-up-a-site-1",
    title: "Hosting a website on the cloud - The Design",
    publishDate: "Nov 14, 2024",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-1.md",
    series: "Setting Up a Site",
    seriesOrder: 1
  },
  {
    id: "setting-up-a-site-2",
    title:
      "Hosting a website on the cloud - Low Level Design and Implementation",
    publishDate: "Nov 20, 2024",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-2.md",
    series: "Setting Up a Site",
    seriesOrder: 2
  },
  {
    id: "setting-up-a-site-3",
    title: "Hosting a website on the cloud - The Database",
    publishDate: "Nov 24, 2024",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-3.md",
    contentType: "swe",
    series: "Setting Up a Site",
    seriesOrder: 3
  },
  {
    id: "setting-up-a-site-4",
    title: "Hosting a website on the cloud: The User Experience",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-4.md",
    publishDate: "Dec 19, 2024",
    contentType: "swe",
    series: "Setting Up a Site",
    seriesOrder: 4
  },
  {
    id: "fearless-rust-write-test",
    title: "Fearless Rust: Write Test",
    videoId: "ng5FK6wkR58",
    mdUrl: getCommon("cdn") + "/target/blogs/fearless-rust/write-test.md",
    publishDate: "Feb 08, 2025",
    contentType: "swe",
    series: "Fearless Rust",
    seriesOrder: 1
  },
  {
    id: "fearless-rust-non-blocking-1",
    title:
      "Fearless Rust: Write a Non Blocking TCP Server From Scratch - Part 1",
    mdUrl: getCommon("cdn") + "/target/blogs/fearless-rust/non-blocking-1.md",
    publishDate: "Feb 13, 2025",
    contentType: "swe",
    series: "Fearless Rust",
    seriesOrder: 2
  },
  {
    id: "iron-learn-1",
    title: " Resuming my journey on learning the basics of AI",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-1.md",
    publishDate: "Nov 18, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 1
  },
  {
    id: "iron-learn-2",
    title:
      " Making my Machine Decide Based on Data",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-2.md",
    publishDate: "Nov 22, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 2
  },
  {
    id: "iron-learn-3",
    title: " The Joy of Running Parallel",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-3.md",
    publishDate: "Nov 23, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 3
  },
  {
    id: "iron-learn-4",
    title: " The Bubble Burst",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-4.md",
    publishDate: "Nov 27, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 4
  },
  {
    id: "iron-learn-5",
    title: "The Comeback",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-5.md",
    publishDate: "Dec 03, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 5
  }, {
    id: "iron-learn-6",
    title: "Building the First Neural Network",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-6.md",
    publishDate: "Dec 06, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 6
  }, {
    id: "iron-learn-7",
    title: "CUDA Integration with Python",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-7.md",
    publishDate: "Dec 31, 2025",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 7
  }, {
    id: "iron-learn-8",
    title: "Proving the Universal Approximation Theorem with Rust",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-8.md",
    publishDate: "Jan 01, 2026",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 8
  }, {
    id: "iron-learn-9",
    title: "Generating Simba Network with Rust",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-9.md",
    publishDate: "Jan 06, 2026",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 9
  }, {
    id: "iron-learn-10",
    title: "The Grand Finale: The Full Image Reconstruction Network from Scratch in Rust",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-10.md",
    publishDate: "Jan 07, 2026",
    contentType: "swe",
    series: "Building a Neural Network from Scratch in Rust",
    seriesOrder: 10
  }, {
    id: "terminal-plotter",
    title: "Writing a Minimal Terminal Plotter",
    mdUrl: getCommon("cdn") + "/target/blogs/writing-terminal-plotter/read.md",
    publishDate: "Jan 22, 2026",
    contentType: "swe"
  }, {
    id: "building-a-word-quiz",
    title: "Building a Real-Time Voice-Driven Word Quiz from Scratch",
    mdUrl: getCommon("cdn") + "/target/blogs/building-a-word-quiz/part-1.md",
    publishDate: "Jun 15, 2026",
    contentType: "swe"
  }, {
    id: "building-a-tab-viewer",
    title: "Building a Web Audio and SVG Synchronized Tab Reader/Writer from scratch",
    mdUrl: getCommon("cdn") + "/target/blogs/tab-viewer-journey/part-1.md",
    publishDate: "Jun 30, 2026",
    contentType: "swe"
  }, {
    id: "building-a-private-cloud",
    title: "Building a Private Cloud: WireGuard, Docker Desktop, and the Silent Linux Kernel Drops",
    mdUrl: getCommon("cdn") + "/target/blogs/private-cloud/part-1.md",
    publishDate: "Jul 17, 2026",
    contentType: "swe"
  }
];

sweContents = modifyArray(sweContents);
