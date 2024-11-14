# Hosting a website online from scratch - Part 1
This will be a multi-part series on how to host a website on internet from scratch. This will be a developer focused series and if you are a developer and want to host your website, keep on reading. You may get some idea of the process.

## Motivation
I have my domain acquired almost a decade now. This is the third time, I am deploying a website on my domain. The last time what I deployed on my domain, was just my Resume. Over the years, I have used some subdomains to showcase some of my past projects. However, they were scattered over places.

This was bugging me for quite a long time. As I was not utilizing the space to its full potential. So, I thought of changing the status quo.

## The requirements
This is the very first step on starting a new project. You have to understand what you want to build. So, I jotted down the things I expect my website to have. To do this, I hopped around various personal sites and tried to find how others have come up with their websites.

These are the list of high level things, I want my website to have:
1. It should have a space for a quick introduction
1. It should have a space for my personal projects to showcase
1. It should have a space to display my notes  on technical learnings and implementations
1. It should have a space for my recently picked up music hobbies
1. It should give me a space to display my job experiences
1. Optionally, I thought of giving a unique touch to my mail address as well
1. Over the years, I could not maintain my last website because, it was quite a chore to put anything on it. So, a big motivator would be get things out of window as easy as possible

## The tools in my arsenal
This is the phase where I took a moment to reflect on what we build and what we already have. This is the phase where ideas should meet the reality.

I already know how to develop web applications using javascript and react.js with some exposure to css. I am not designer or hard core front end developer. I am a back end heavy full stack developer. So, I would be able to start up the basics and can pick up things on the way.

Also, I don't aspire to build a highly sophisticated graphics heavy or too interactive 3d.js or other library website. My requirements are simple, so a simple layout should do fine.

## The design
Once I jotted down the basic requirements and checked the tools, I started thinking of how to implement.

The very first thing that comes to mind is the layout. How the web page should look like. May not be to the exact point, but some rough design work should be done at first, then you can go for design.

For my case, I had some idea of having 4 pages in my website, that I want to host. I drew them on the page to think around the next step. Following is what I came up in the first draft. The right now website is almost the same one, except that few things have been added and shifted little here and there.

![Home Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/home-page.png "Home Page Draft")
1. The top most bar will be navigation bar
1. The bottom most bar is footer with copyright and my name
1. The middle section will be two rows, each having two columns
1. The first row left column is some header with my name followed by a paragraph of short description
1. The first row right column will be my photo
1. The second row left column, I will have a short list of my technology blogs
1. The second row right column, I will have a short lisr of my music blogs

![Blog Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/blog-page.png "Blog Page Draft")
1. The blog page will have a header, some generic text and an introduction
1. The next section will be a two column display, on the left will reside all the tech blogs and on the right, all the music related contents

![Projects Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/proj-page.png "Project Page Draft")
1. Like the blog page, project page will also have a header and some intro
1. Then as the main body, I will keep all projects as some kind of block display.
1. Each project block will have a header, a body which will hold the short description of the project
1. A Technology icon like Java, Rust, React etc.
1. In the footer, there will be a link or button which will redirect me to the project's github page

![About Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/about-page.png "About Page Draft")
1. The about page is what I hosted for quite long in the past, my resume
1. It will have all my past experience
1. All the technology stacks, I have worked with
1. All the relevant professional details etc.

This was the first draft of my design. However, as the development process kicked in, I also realized, I can make it more interactive and added few more features, like adding an icon in each project tile to display the project's README.md file or a Game Console Button to play the game on the website itself etc.

