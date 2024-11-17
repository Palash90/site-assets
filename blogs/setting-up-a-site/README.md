1. Develop the application
    - Focus on Markdown display
    - Home page complexity
1. Configure and design the data source
1. Make the website as dynamic as possible with static hosted js
1. The mobile display
    - React Bootstrap was a life saver
1. Catch eyes with different fonts
    - Font types
1. Color
    - Color Wheel and psychology 
1. Custom color pallete with bootstrap css variables
    - Did not work.
    - I made the markdown css
    - Then I created common variables for all the header colors. My page does not have much of various elements to color for
    - Then moved and created my own variable set for markdown css. 
    - Created one more theme
1. Dynamically generate the website source data using git pre-commit. This is exceptionally powerful.
1. Make a sitemap
1. Style the sitemap
1. Now during publishing of website, I faced with an issue. I was copying and pasting image links left and right however. I found it cumbersome to copy paste and then there is this chance of maintainability issue. What if I later move to some other system for hosting static site assets?
    - I first thought of using markdown variables. However, there is not support for variables. You can use some kind of link referencing in markdown. However, that did not work properly when I tried to embed a link inside a link component or image component of markdown.
    - Then I came up with this solution. I tweaked the markdown rendering component in the app. Then I put all my links to the js file with unique ids, then in the app, I got the key and then applied the value in the rendered component.
1. Static hosting
    - Issue with github.com routing protocol

1. Using Termux to change files on the go.
1. The power of acode and its acodex terminal
1. Using Google analytics
    - Use of ReactGA.pageview
    - Fixing it with ReactGA.send
1. check
1. Just like that, mobile edit

