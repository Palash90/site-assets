# Hosting a website on the cloud: The User Experience

## Going Online
Excited with dynamic data flowing in the site and seeing all the blogs correctly, I finally marked the completion checklist. 

Then, I thought, let's host this. Like, I hosted the site-assets website, I also hosted this site.

With a hope, I clicked on a blog. A sudden heartbreak occured. I was thrown back to the home page instead of blog.

## Identifying the issue
This is new for me. I cross verified from `localhost`. Yes, it works as expected, all my routes are working when launched from `localhost`.

I retested every link of the website and ensured everything is working perfectly. This made me pin point the issue on `github pages`.

## The resolution
I started learning about how github pages work and what I can do about it. The issue boils down to how github pages handles routes and came up with a solution of usinng `HashRouter` component of react instead of `BrowserRouter`.

Accordingly, made changes to all the links to have a `#` which gives the control back to the application, rather than the way `Github Pages` handles routes.

Redeployed the site. Now, it started working.

## The Look and Feel
Excited, I started poking around the site from various devices - Mobile, Tablet, TV, Computer.

I found me hating myself to inflict such a pain to look at the website. I am not a designer, still it felt very awkward to even stick for 10 seconds in the site.

All alignments are gone for a toss, contents are showing hapazardly, my profile picture is showing weird. To top it off, I had a `table` component placed in between the page, which rendered at the furthest corner and messed up everything.

I started thinking about the problem. I have worked with bootstrap earlier. So, the solution came easily. I used a few layout component, an image holder and a grid layout.

With these only I fixed the major issues and at least the pages were loading decently on every device I could get hold of.

## Approaching colors
At this point, the site was live, looking somewhat ok-ish. At least I can see things aligned properly and readable.

Still, something was missing. I quite could not figure out what. It was giving me an itch. I took an hour brrak to get a fresh perspective. Still it did not resolve the issue.

At this point, I ccalled my wife and daughter to take a look. at a glance, they revealed the problem to me, "Too Gray", "Too Bland", "looks like news print of 19th century", "no one will read it". These were the comments. I am still not sure, how many will read my blog after new design. But yeah, point taken, no harm in trying.

Honestly, I didn't even know a bit of design. Luckily, while working in GE, I got tasked to work with design team for few days. At that, time I had purchased a Udemy Course on UX Design. It is quite longng course. I did not go into all the aspects of it. I just focused on the typography and color section.

In design, there is a proven concept of color wheel and you can make different color schemes based on the color wheel. Then I started finding colors for my site. To do that, I knew about color psychology. Each color portrays a mood, a message.

After going through few materials on this, I found that `Tech Blue` is a color that has been used in many applications, especially for technological aspects. I chose the same. Now, I had to decide the color pallette. I simplified the process, chose a monochromatic pallette with 5 colors.

I distributed the colors in different sections of the site like the header, the body, the links etc. Also, while choosing color pallettes, I had to keep an eye pn the contrast ratio too. With a small contrast ratio, the text will be hard to read, while too high a ratio is going to be too contrasting. I had to strike a balance there. Although, its not perfect but it works. May be, in future I can revisit the colors and play with the color pallette.

## The Typography
When it comes to the UI, there are many things that matter. One such element is typography. 