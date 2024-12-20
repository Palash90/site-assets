# Hosting a website on the cloud: The User Experience

## Going Online
Excited with dynamic data flowing on the site and seeing all the blogs correctly, I finally marked completion in the checklist. 

Then, I thought, let's host this. Like, I hosted the site-assets website, I also hosted this site.

With a hope, I clicked on a blog. A sudden heartbreak occurred. I was thrown back to the home page instead of showing the blog.

## Identifying the issue
This was new for me. I cross-verified from `localhost`. That worked as expected, all routes are working when launched from `localhost`.

I re-tested every link of the website and ensured everything is working perfectly. This made me pin point the issue on `GitHub Pages`.

## The resolution
I started learning about how GitHub Pages work and what I can do about it. The issue boils down to how GitHub Pages handle routes and I came up with a solution of using `HashRouter` component of react-router instead of `BrowserRouter`.

Accordingly, made changes to all the links to have a `#` which gives the control back to the application, rather than the way `Github Pages` handles routes.

I redeployed the site with these changes, and it started working.

## The Look and Feel
Excited, I started poking around the site from various devices - Mobile, Tablet, TV, Computer.

I found myself hating the pain I inflicted bt looking at the website. I am not a designer, still it felt very awkward to even stick for 10 seconds in the site.

All alignments were off, contents were showing hapazardly, and my profile picture looked weird. To top it off, I had a `table` component placed in between the page, which rendered at the furthest corner and messed with every other component.

I started thinking about the problem. I have worked with bootstrap earlier. So, the solution came easily. I used a few layout components, an image container and a grid layout.

With these only I fixed the major issues and at least the pages were loading decently on every device I could get hold of.

## Approaching colors
At this point, the site was live, looking somewhat okay-ish. At least I can see things aligned properly and readable.

Still, something was missing. I couldn't quite figure out what. It was giving me an itch. I took an hour-long break to get a fresh perspective. Still, I could not figure out the root cause.

At this point, I was frustrated and decided to outsource the problem to my wife and daughter. At a glance, they revealed the problem to me, "Too Gray", "Too Bland", "looks like news print of 19th century", "no one will read it". These were the comments. I am still not sure how many will read my blog after the new design. But yeah, point taken, no harm in trying.

Honestly, I didn't even know a bit of design. Luckily, while working in GE, I got tasked to work with design team for a few days. At that time, I had purchased a Udemy Course on UX Design. It is quite a long course. I did not go into all the aspects of it. I just focused on the typography and color section.

In design, there is a proven concept of the color wheel, and you can make different color schemes based on the color wheel. Then I started finding colors for my site. To do that, I had to know about color psychology. Each color portrays a mood, a message.

After going through few materials on this, I found that `Tech Blue` is a color that has been used in many applications, especially for technological aspects. I chose the same. Now, I had to decide the color pallette. I simplified the process and chose a monochromatic pallette with five colors.

I distributed the colors in different sections of the site like the header, the body, the links etc. Also, while choosing color palettes, I had to keep an eye on the contrast ratio too. With a low contrast ratio, the text will be hard to read, while too high a ratio is going to be too bright. I had to strike a balance there. Although it's not perfect but it works. Maybe in the future I can revisit the colors and play with the color palette.

## The Styling Variables
Changing colors in different components started becoming a headache pretty soon. I started thinking of reducing the effort here. There are ways with SCSS, but I did not want that as the site is not heavy with colors and custom styles. Most of the pages, the style is almost consistent. So, I just needed a way to make a handful of variables.

The very first thing I tried was `CSS variables`. Well, that did not work with Bootstrap. Then I played with different stylesheets and class names. That fourtunately worked.

However, there still was this problem of changing every color in all the stylesheets. After looking into this issue for a few hours, I came up with a set of variables, that I can use, and I could use a set of 7 variables that will suffice my need for color schemes.

Then, the whole coloring scheme boiled down to just selecting seven different colors. That's how the whole color scheme of this whole website is working on.


## The Typography
When it comes to UI, many things that matter. One such element is typography. Till this point, the site used whatever default Bootstrap came up with.

But while going through the section on typography, I got to know about different types of fonts and how to use them effectively. I again started mixing fonts from different types. Some were good, some bad and some really ugly.

Finally, after playing with different fonts for a day, I settled on the most widely used `Merriweather`, `Lora`, `Roboto` and `Open Sans`. The site uses combinations of these in different types of contents.

