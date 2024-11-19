# Hosting a website online from scratch - Low Level Design and Implementation
Once I was done gathering the requirements, followed by designing the web app and choosing libraries, it was time for me to dive deeper into the build process.

The first thing that hit me was the way to handle different pages. I wanted all the pages to have a distinct url, so that it can be bookmarked. The very first thing that I came up with is the routes that I will have in my website. I started with 4 urls initially for each of the pages in the design phase. Then, for each of the pages, I started thinking, if I can make any sub-page for it.

Then I divided each page into sub-pages and sections.

## Navigation Links
When the webapp is hosting multiple pages, a user needs a way to move back and forth between the pages. For this purpose, I put all my main links, namely `home`, `blogs`, `projects` and `about` in a navigation header component. When I was doing that, I thought of also putting in a footer with a copyright icon. Simple navigation header and footer implemented.

To now navigate between these links, I used `react-router` which displays different page based on user clicked link.

## Home Page
Home page did not have much change. It was already designed to have 4 different sections, which would be easy to implement by using the `react-bootstrap` layout. Neither of the sections needed its own page, except for the blog links. Which I anyways have to deal in the next page, so I deferred it till that point. Hence, only one url would suffice for the home page - `/`. I quickly got a picture of mine, some text and some arbitary placeholder for the blog links. Voila, the home page of my website was born.

## Blogs Page
The second page was the `blogs` page, arguably the most difficult to implement. Initially, I made the design to have two sections side by side having a list of blog links. But then it hit me, I have two types of blogs to be delivered. May be a music blog reader would not be much interested in the tech blog and vice versa. Also, if I take two different routes, for music and tech blogs, I can host them seaparately on two sub-domains. Hence, I made the decision of separating the blogs page in two - `Tech Blogs` and `Music Blogs`.

My work was not over on this page. I had to find a way to publish each blog on its own url.

Keeping all these points in mind, I created  several routes for the content page -
1. The main content pages
   - `/contents/music`
   - `/contents/tech`
1.  The individual blog pages like `/content/vi-essentials`, `/content/vi-essentials` etc. 


The main `blogs` page design was fairly simple, a simple header, some introduction and a list of elements. Easily achievable by some form of `header`, `paragraph` and `list` component

The next big task was to actually think of how to store and display the blog pages. For display, I already figured out to use `markdown` as my choice. The question remains how to store the files so that, I can easily create, update or edit contents and subsequently update the lists of blogs and the updated blog itself without me touching the main code base.

The solution I used, was to host a javascript file with all the relevant blog details. I came up with the following structure for the same.

  ```json
  {
          id: "grafana-on-aws",
          title: "Configure Grafana on AWS",
          contentType: 'tech',
          videoId: "",
          mdUrl: "/target//blogs/aws-grafana/README.md",
          publishDate: "Jun 11, 2024"
  }
  ```
  - `id` will give the url an unique id to host the page using content id url parameter
  - `title` will give the short one liner description of the blog
  - `contentType` field hints if it should display this in `music` or `tech` blog
  - `mdUrl` is the actual url of the hosted markdown file, we can load content of this file in `useEffect` hook
  - `videoId` is for those blogs where I need a video to be also published with it. Mostly used in guitalele tutorials.
  - `publishDate` is to display the date of publish on the blog page. 
      
Once all these were figured out, the rest was easy to implement. I just have to fetch the content of `mdUrl` on the blog page and wrap it up in `react-markdown` component for blogs or for youtube videos, simply use the page embed or for contents which has both the elements, a react layout container containing both.

## Projects Page
 Once the blog page was implemented, I moved to figure out how to build the `Projects` page. This page was comparatively easy to implement. I already figured out, I will be using `Card` elements from `react-bootstrap` and will use `react-icons` to show the technology specific elements, the dynamic data source problem was already solved during the design and implementation of the `blogs` page. All I had to do during the implementation of this page was to reuse already implemented components for this page specific layout. That's it. I came up with the following json structure for this page and put elements in the page according to the design.
```json
{
        id: "hdl-emulator",
        name: "HDL Emulator",
        desc: "An online Web Based HDL Emulator for Teaching Purpose",
        type: "javascript",
        playUrl: "https://emulator.palashkantikundu.in/",
        githubUrl: "https://github.com/Palash90/emulator",
        mdUrl: getCommon('raw_content_cdn') + "/emulator/refs/heads/main/README.md"
}
```
  - `id`: A unique id
  - `name`: Name of the Project
  - `desc`: A short one liner description on what the project does.
  - `type`: Technology used to build the project, used as an icon on the project card.
  - `playUrl`: If the project is hosted somewhere, direct the user to the page
  - `githubUrl`: Project repository url
  - `mdUrl`: Not used initially but then idea struck to use it.

Once it was built, I thought of enhancing the web page. The first modification I made is to use the `mdUrl` field. I already have a blog rendering component which is nothing but a `markdown` viewer, I can definitely use that component here as well. I used it with a `bootstrap modal` and an info icon to show the project's `README.md`.

The second modification was to actually take the existing react based web apps from other projects like `tic-tac-slide` and merge with the personal website project and directly render in the same web app. For projects like this, I used a game controller icon to hint the user to play the game.

If you are still here, it is good time to take a look at the `Projects` page [here](/#/projects) to get a visual feel of the implementations.

## About Page
When I reached this point, it was clear that, I don't need any special page for this, I can simply resuse the `Blog Page` and host a blog with all my details and wrap that blog in the `about` page. That's it.

The web app is ready, all pages are ready, all navigation links are functional. I wrote a page with all my details, took some of my old blogs and packaged it in `markdown` format, took a few youtube videos and embedded them in the `data.js` file and wore the QA hat.

I felt something important missing from it. Took to others and it was easily pointed out - social media and contact details are missing from the website. I took few `react-icons` and packaged them in a component side by side and placed it in both the `home` page and in the `about` page.

Finally, the webapp is ready to roll. You can find the source code of this website here - [Source Code](https://github.com/Palash90/site)
