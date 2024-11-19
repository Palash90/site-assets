# Hosting a Website Online from Scratch: Low-Level Design and Implementation

Once I was done gathering the requirements, designing the web app, and choosing the libraries, it was time to dive deeper into the build process.

The first thing I focused on was how to handle different pages. I wanted all pages to have distinct URLs so they could be bookmarked. I started by defining the routes I would need for my website. Initially, I created four URLs, one for each page planned during the design phase. Then, for each page, I considered whether it could have sub-pages.

I divided each page into sub-pages and sections for better organization.

## Navigation Links

When a web app hosts multiple pages, users need a way to navigate between them. To address this, I placed all the main links— `Home`, `Blogs`, `Projects`, and `About` in a navigation header component. While doing this, I also decided to add a footer with a copyright icon and my name. A simple navigation header and footer were implemented.

To enable navigation between these links, I used `react-router`, which displays different pages based on the link clicked by the user.

## Home Page

The `Home` page didn't require much change. It was already designed to have four different sections, which were easy to implement using the `react-bootstrap` layout. None of the sections required their own page, except for the blog links, which I planned to handle on the `Blogs` page. As a result, a single URL sufficed for the Home page: `/`. I quickly added a picture of myself, some text, and placeholder content for the blog links. Voilà! The Home page of my website was born.

## Blogs Page

The Blogs page was the second page I tackled and, arguably, the most complex to implement. Initially, I designed it to display two sections side by side, each containing a list of blog links. However, I realized that I had two distinct types of blogs: music and tech. A music blog reader might not be interested in tech blogs and vice versa. Moreover, creating separate routes for music and tech blogs allowed me to host them on separate subdomains. Thus, I decided to split the `Blogs` page into two: `Tech Blogs` and `Music Blogs`.

But the work didn’t end there. I also needed a way to publish each blog on its own URL.

Considering these points, I created several routes for the content pages:

1. Main Content Pages
   - /contents/music
   - /contents/tech
1. Individual Blog Pages. For example, /content/vi-essentials or /content/aws-grafana.


The design of the main Blogs page was straightforward: a header, some introductory text, and a list of blog links. This was easily achieved using `header`, `paragraph`, and `list` components.

The next big task was figuring out how to store and display the blog pages. I had already decided to use `Markdown` for the blog content. The question was how to store the files so that I could easily create, update, or edit content and automatically update the list of blogs without touching the main codebase.

The solution I implemented was to host a JavaScript file containing all the relevant blog details. Here’s the structure I used:

```json
{
  id: "grafana-on-aws",
  title: "Configure Grafana on AWS",
  contentType: "tech",
  videoId: "",
  mdUrl: "/target/blogs/aws-grafana/README.md",
  publishDate: "Jun 11, 2024"
}
```
  - `id`: A unique identifier for the blog, used to generate the URL.
  - `title`: A short description of the blog.
  - `contentType`: Indicates whether the blog belongs to the music or tech category.
  - `mdUrl`: The URL of the hosted Markdown file. This file’s content is loaded using the useEffect hook.
  - `videoId`: For blogs that include an accompanying video (e.g., guitar tutorials).
  - `publishDate`: The blog’s publication date.

Once this structure was in place, the rest was straightforward. I fetched the content of `mdUrl` on the blog page and displayed it using the `react-markdown` component. For blogs with videos, I embedded the video. For blogs with both Markdown and videos, I used a `react-bootstrap` layout container to display both elements.

## Projects Page

After completing the Blogs page, I moved on to the `Projects` page, which was relatively easier to implement. I decided to use `Card` elements from `react-bootstrap` and `react-icons` to display technology-specific icons. The dynamic data source problem had already been solved during the `Blogs` page implementation, so I reused the same approach. Here’s the JSON structure I used for the `Projects` page:

```json
{
  id: "hdl-emulator",
  name: "HDL Emulator",
  desc: "An online web-based HDL Emulator for teaching purposes.",
  type: "javascript",
  playUrl: "https://emulator.palashkantikundu.in/",
  githubUrl: "https://github.com/Palash90/emulator",
  mdUrl: getCommon('raw_content_cdn') + "/emulator/refs/heads/main/README.md"
}
```

  - `id`: A unique identifier for the project.
  - `name`: The project’s name.
  - `desc`: A brief description of the project.
  - `type`: The technology used, displayed as an icon on the project card.
  - `playUrl`: A link to the live project, if hosted.
  - `githubUrl`: The project’s GitHub repository.
  - `mdUrl`: A URL to the README file for additional details.

To enhance the page, I reused the Markdown viewer from the Blogs page to display the project’s `README` file in a modal when an info icon is clicked. Additionally, I integrated React-based web apps from other projects, `like tic-tac-slide`, directly into the website. For such projects, I used a game controller icon to hint that the project is playable.

If you are still here, it is good time to take a look at the Projects page [here](/#/projects) to get a visual feel of the implementations.

## About Page

By this point, it was clear that the `About` page didn’t need to be a separate design. I reused the Blog Page component to host a blog containing all my details and wrapped it within the About page.

The web app is ready, all pages are ready, all navigation links are functional. I wrote a page with all my details, took some of my old blogs and packaged it in markdown format, took a few youtube videos and embedded them in the `data.js` file and wore the QA hat.

I felt something important missing from it. Took to others and it was easily pointed out - social media and contact details are missing from the website. I took few `react-icons` and packaged them in a component side by side and placed it in both the home page and in the about page.

Finally, the webapp is ready to roll. You can find the source code of this website here - [Source Code](https://github.com/Palash90/site)


