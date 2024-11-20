# The writing process
The writing process involves simple steps. I have automated most of the processes, however some processes are still not working as expected. Hence, I have to keep all the writing process handy.

Follow this process for general blog writing process.

## Pre-requisite
1. If you are on desktop, use `VS Code` and clone the `site-assets` repository
1. Then in the `.git/hooks` directory, add a `pre-commit` file with just one line - `sh pre-commit.sh`
1. The majority of work is done in `pre-commit.sh`  file
1. If your are using mobile device, install `Acode` and `Termux`. On `Termux`, you should install `acodex-server`. Check online you will get to know how.
1. On `Acode` editor, install the `Acode Terminal` extension. This extension uses afore-mentioned `acodex-server`. This should be running in termux.

## The Writing process
1. Create a new folder under `blogs` folder in this repository with the name of the blog. This usually can serve the purpose of making the blog id as well. Then add your `markdown` file inside this directory.
1. Then add a json object in the `drafts.js` file under `site-contents/js` and commit the changes.
```
{
        id: "writing-process",
        title: "How to write a new blog",
        mdUrl: getCommon("cdn") + "/target/blogs/writing-process/README.md",
        publishDate: "Nov 08, 2021",
        contentType:"swe"
}
```
1.  This will enable you to see the draft version on the website, which will be in this location of the site- `/#/content/:contentId`
1.  If you are on PC, usually the `pre-commit` hook works. However, if you are on mobile devices like mobile or tab and using `Acode` and `Acode Terminal`, you should run `sh pre-commit.sh` from the root directory of this repository.
1. Once these steps are done, simply commit the changes and push to the repository. Then wait for the deployment to complete and check on the site. The blog should be available to you only. There will be no link on the main pages of the site.

## Adding images or other links
1. If you want to host images or any other file links on your markdown, there is a process to do that too.
1. If it is an internal link, simply use the relative path. For example, if I want to link to static file hosting page, a simple way to add this relative link is this - [Static File Hosting](/#/content/static-file-hosting)
1. If you want add an image to your blog, it's a bit more involved
    1. First add the image to the blog directory.
    1. Then add a new key-value pair in the `variableMap` of `site-contents/change-markdown.js`
    1. Then use the `key` in the image tag of the `markdown`
    1. During processing of the repo before commit, the `markdown` files will be changed to actual url of the image.
    1. For example, I have added the following image with this variable `${`static-file-hosting-github-pages`}` The image tag is this - `![Github Pages enabled](${`static-file-hosting-github-pages`} "Enable github pages")`
    ![Github Pages enabled](${static-file-hosting-github-pages} "Enable github pages")
