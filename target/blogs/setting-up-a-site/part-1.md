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
This is the phase where I took a moment to reflect on what I want to build and what I already have. This is the phase where ideas should meet the reality.

I already know how to develop web applications using javascript and react.js with some exposure to css. I am not a designer or hard core front end developer. I am a back end heavy full stack developer. So, I would be able to start up the basics and can pick up things on the way.

Also, I don't aspire to build a highly sophisticated graphics heavy site or employ too interactive 3d.js or other library website. My requirements are simple, so a simple layout should do fine. Often a minimalistic approach helps a lot to kick start the project.

## The design
Once I jotted down the basic requirements and checked the tools, I started thinking of the implementation.

The very first thing that comes to mind is the layout. How the web page should look like. May not be to the exact detail, but some rough design work should be done at first, then you can go for a more detailed approach.

For my case, I had initial idea of having 4 pages in my website, that I want to host. I drew them on the page to think around the next step. Following is what I came up in the first draft. The right now website is almost the same one, except a few things have been added and shifted here and there, not vastly changing the intial design.

![Home Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/home-page.png "Home Page Draft")
1. The top most bar will be navigation bar
1. The bottom most bar is footer with copyright and my name
1. The above two elements will be applicable to all the pages
1. The middle section will be two rows, each having two columns
    - The first row left column is placeholder for an introduction
    - The first row right column will display my photo
    - The second row left column, I will have a short list of my technology blogs
    - The second row right column, I will have a short list of my music blogs



![Blog Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/blog-page.png "Blog Page Draft")
1. The blog page will have a header, some generic text and an introduction
1. The next section will be a two column display
    - The left will show all the tech blogs 
    - The right, all the music related contents



![Projects Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/proj-page.png "Project Page Draft")
1. Like the blog page, project page will also have a header and some intro
1. Then as the main body, I will keep all projects as some kind of block display.
    - Each project block will have a header, a body which will hold the short description of the project
    - A Technology icon like **Java, Rust, React** etc.
    - In the footer, there will be a link or button which will redirect me to the project's github page



![About Page Draft](https://palash90.github.io/site-assets/blogs/setting-up-a-site/about-page.png "About Page Draft")
1. The about page will be an online version my resume
1. It will have all my past experience
1. All the technology stacks, I have worked with
1. All the relevant professional details etc.


This was the first draft of my design. However, as the development process kicked in, I also realized, I can make it more interactive and added few more features, like adding an icon in each project tile to display the project's **README.md** or a Game Console Button to play the game on the website itself etc.

## Figuring out the architecture
Once these details fell into place, the very next thing I had to decide was how to build the system part by part. Following is how it went.
1. I was pretty sure, I will be using React.js for this task. So, that was easy.
1. The next thing, the library that I am going to use to beautify the pages. There are many competitive players in this area - Bootstrap, MDB etc. I chose Bootstrap and React Icons; mostly these two libraries will suffice all my needs for Reusable Components and handy dandy icons.
1. For each of the paths, I would also need router. So, react-router would be my next choice for library.

The next big challenge in the thinking process was to identify how to store and fetch the blogs. For this, I had to do some research.

I was going through my old blogs. I realized, I can move all my blogs to Markdown format. It suddenly occurred to me that, almost all the pages can be markdown, except for the static four. Within that too, I recognized that **About** page can also be rewritten in markdown. So, if I can support markdown rendering, I am done almost 90%.

The **Blogs** page will be a plain list of items. **Projects** is also a list except that the rendering will done on box like component.

At this point, the choice was easy. **react-markdown** is a great fit for my use case.

With this thing detailed out, the question remains, how and where to store the markdown files?

It is easy to go with the thought process and at the end of 2 hours you may end up essentially a whole cloud stack inside your head but nothing on paper. On top of that, I was pretty reluctant to setup a database, a server, some backend Java/node/python code etc. So, I cut short my thoughts there. I have a pretty basic use case to take care and scalability can take a back seat for now.

Storing raw markdown files somewhere easily accessible was the topmost choice for me.

With that figured out, the next part of the guessing game was to actually figure out, where to store the static markdown files. The answer was always in front of me, I just could not recognize it in its plain form. It's **Github**, a perfect cloud storage system available for free and almost zero hassle getting data out of it. All I had to do was to setup github pages for one of my repositories.

I was sure about one more thing as well. I am not going to write a single hard coded text on the React components, anything that goes to display, should come from a json. The more, the merrier.

That was pretty much the design of the site. In the next one, I will show you how to set things up.
