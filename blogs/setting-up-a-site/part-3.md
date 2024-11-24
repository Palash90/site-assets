# Hosting a website on cloud: The database

**I don't have a backend to handle data manipulation for my site. Then how do I publish new blog or edit an existing one?**

In our [last post](/#/content/setting-up-a-site-2), we saw how we could develop a web app. We have successfully built all the pages, set up router information and came up with a way to display markdown and youtube videos. That makes the frontend complete. However, if you review the process, you can see that, if we need to change anything on the website(like publishing or editing a blog post), we cannot do that without redeploying our webapp, In a nutshell, the webapp is not dynamic.

In this post, we will go over the path I took to make the static webapp dynamic using some tips and tricks. I was clear that I am going to use GitHub Pages for this one too. So, I started with creating a new repo which will hold the static resources, separate from the main website repo.

## Separation of Concern

I separated `script.js` and `styles.css` from the react app and hosted these two files on github pages, through a different repository and enabled hosting from `main` branch. If you want to read details on how to do it, read [here](/#/content/static-file-hosting).

At this point, I was able to change my website pages by simply making changes on the second repo, without the need to redeploy. This enabled clear separation of tasks. Frontend is only responsible for rendering the web page. The second repo, mimics the backend as it handles the data layer.

## Handling the Maintenance Mess
Maintaining all data in a huge JS file quickly became a nightmare. Finding out places to change 1 line data in a 3000 line javascript file became a pain. I started wondering if somehow I can break this huge file into manageable sections and write another script to merge all these pieces into one JavaScript file. This is exactly how things are done. Take a look at the following files.
```
$tree
|_ techBlogs.js
|_ drafts.js
|_ musicBlogs.js
|...
   
```
Once I am done with editing or writing, I run one script before committing my changes. The `node` script consolidates all these files into a single one. When all files are uploaded and deployed on GitHub pages, I get all the required data to run the site smoothly. Soon, it hit me, I can do the same with Stylesheets as well. So, I broke down the stylesheets into different pieces and during the consolidation phase, I was merging them all. Not only that, I also made a unique key constraint check for all the blog entries. So that, each blog has a unique identifier to use in the frontend to render the correct blog. Following is an entry from the blog list.
```
     {
        id: "setting-up-a-site-3",
        title: "",
        publishDate: "Nov 20, 2024",
        mdUrl: getCommon("cdn") + "/target/blogs/setting-up-a-site/part-3.md",
        contentType:"swe"
    }
```
The `script` now works as a `key-value` database.

I attached a recursive map traverse function along with the consolidated data which on the frontend, I can use for selecting the key I want.

```
function findProp(prop) {
    var obj = data
    prop = prop.split('.');
    for (var i = 0; i < prop.length; i++) {
        if (typeof obj[prop[i]] == 'undefined')
            return null;
        obj = obj[prop[i]];
    }
    return obj;
}
```

In a nutshell, the JavaScript file is working as a data source for frontend and this consolidated data is generated through a script, much like replicating the `INSERT`, `UPDATE` and `DELETE` statements of a database while the `SELECT` queries are being handled in frontend using the map traverse function.

## The automation trick
Once I achieved this, I found I missed running the script. Then I figured out, this can be easily handled by a `pre-commit` script during `git commit`. I used this trick to automate the whole process.

## The simplified process for publishing
Now all I need to do is to write a blog in familiar, easy to write and version controlled `markdown` format, add the entry to a javascript file and commit. A short phase of waiting and I see my blog live on my site. In fact, this blog you are reading right now, has been published usung the same process.

## Source Code
Source code for the second repo can be found [here](https://github.com/palash90/site-assets)
