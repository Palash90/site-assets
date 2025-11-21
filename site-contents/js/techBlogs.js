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
  },
  {
    id: "setting-up-a-site-2",
    title:
      "Hosting a website on the cloud - Low Level Design and Implementation",
    publishDate: "Nov 20, 2024",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-2.md",
  },
  {
    id: "setting-up-a-site-3",
    title: "Hosting a website on the cloud - The Database",
    publishDate: "Nov 24, 2024",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-3.md",
    contentType: "swe",
  },
  {
    id: "setting-up-a-site-4",
    title: "Hosting a website on the cloud: The User Experience",
    mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-4.md",
    publishDate: "Dec 19, 2024",
    contentType: "swe",
  },
  {
    id: "fearless-rust-write-test",
    title: "Fearless Rust: Write Test",
    videoId: "ng5FK6wkR58",
    mdUrl: getCommon("cdn") + "/target/blogs/fearless-rust/write-test.md",
    publishDate: "Feb 08, 2025",
    contentType: "swe",
  },
  {
    id: "fearless-rust-non-blocking-1",
    title:
      "Fearless Rust: Write a Non Blocking TCP Server From Scratch - Part 1",
    mdUrl: getCommon("cdn") + "/target/blogs/fearless-rust/non-blocking-1.md",
    publishDate: "Feb 13, 2025",
    contentType: "swe",
  },
  {
    id: "iron-learn-1",
    title: "Resuming my journey on learning the basics of AI",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-1.md",
    publishDate: "Nov 18, 2025",
    contentType: "swe",
    draft: true,
  },
  {
    id: "iron-learn-2",
    title:
      "93% Accuracy and a System Crash: My Journey with Logistic Regression",
    mdUrl: getCommon("cdn") + "/target/blogs/iron-learn/iron-learn-2.md",
    publishDate: "",
    contentType: "swe",
    draft: true,
  }
];

sweContents = modifyArray(sweContents);
