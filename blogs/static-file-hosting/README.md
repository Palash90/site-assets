# Hosting Static Web Site Assets

Recently, I thought of revamping my whole website. The domain just hosts my education and work experience. Nothing else. It is just a resume hosted on website. However, apart from that, I could use the space for my projects, blogs etc. They are on the internet however, they are not under the same umbrella. Everyone has their own sub-domain. Thus, becoming very hectic to maintain over time.

## Objectives
- Most of the website can be designed using only static pages
- I should be able to host content easily without deploying the web application once again
- Use only freely available tools

## Challenges
- You either need to have some kind of cdn to deliver your content
- however, I could not think of a simple solution

## Solution
- Use Github pages to host your static files
- Use these resources in your application

## Steps
- First you need to have a github account
- You need to create a dedicated repository to host your files
- You need to enable github pages to do that
  ![Enable Github Pages](${static-file-hosting-github-pages} "Enable Github Pages")
- Upload your files on the repository
- Access it by following the directory structure
  ![Access JS Files](https://palash90.github.io/site-assets/blogs/static-file-hosting/js-hosting.png "Scripts")
  ![Access Images](https://palash90.github.io/site-assets/blogs/static-file-hosting/image-hosting.png "Images")
- Now use these resources in your Markdown files
- Use some kind of Markdown renderer, like the one you are reading this on
- Voila, you have your blog website ready

## Limitations
- The biggest limitation of this approach is, you don't get a way for reader interaction. Such that the reader can post comments on your blog. If you have any solution for this, please let me know.