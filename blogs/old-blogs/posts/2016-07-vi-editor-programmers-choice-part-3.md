---
title: "vi editor: Programmers' choice - Part 3"
date: "2016-07-14T12:27:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045038)"
---

Now we have a pretty large file containing 150 lines and 6450 characters. It will surely go beyond a single screen to display the file. At this time, this may be a difficult job to read the file.

We can try **h j k l combination to read the file but it is indeed pretty awkward to do so.

To save us from this manual and irritating job, vi has blessed us with some more cool commands which we will try here. Again remember all of this post works in Command Mode and **not in Insert mode.

First, open the file in vi editor and use **:set number to show the line numbers.

Let's start to jump on a particular line in this large text file.
**G - moves to the last line of the file. No matter how long the file is, it will do its job pretty quickly.

Once you are in the last line, you may need to go to the beginning of the file or anywhere in between. So, let's try this one by one.

**1G - moves the cursor to the first of the file
**20G - moves the cursor to the first character of line  number 20
**30G - moves the cursor to the first character of line  number 30

So, you get the idea, <number>G will move the cursor to line number <number>. While this is very handy, we can do the same using the following command as well.

**:1
**:20
**:30

So **:<number> will move the cursor to the beginning of line number <number>.
Now what if you've used a number that is larger than the total number of lines in the file (for our file we can try 160, 170, 1000 etc.).

Actually this will take you to the beginning of the last line.

So, in our case **160G, **170G, **100G, **G, **:160, **:170, **:1000 yields the same result.

Next we want to move through line. Following are some commands that will help you.

**- (hyphen) - moves the cursor to the beginning of the previous line.
**+ - moves the cursor to the beginning of the next line.

In many cases, you may not know the exact position of a particular portion of text and you also don't want to scroll through line by line or you are reading a large file and you want to move through screens. These are pretty common scenario when you are reading a log file or reading a beautiful long story in a text file. In these scenarios, you simply want to go to the next screen or previous screen.

vi comes with four commands

**Ctrl-f scrolls down one screen
**Ctrl-b scrolls up one screen
**Ctrl-u scrolls up a half a screen
**Ctrl-d scrolls down a half a screen

This one really needs some screenshots to understand. Let me show you.
Initially I am at the beginning of first line and my screen contains 41 lines of text at once.
![image](https://2.bp.blogspot.com/-LvT3qIQmGYA/V4j-qgm6YqI/AAAAAAAACuk/p3Cx-VHio40LCpCQHvpp4DWcmnmecurWgCEw/s1600/Screenshot_20160715_204632.png)

Now I am entering **Ctrl + f 
![image](https://3.bp.blogspot.com/-RvpZoTDsikY/V4j_MryFTwI/AAAAAAAACuk/291wESxtBhsLa9ryGPgOc11TLP4_pr_4gCEw/s1600/Screenshot_20160715_205001.png)

Now my screen contains lines 40 to line 80. This screen contains two lines from the previous one to have some trail of text.

Other combination should be tried by you on your own. Also note that, the lines on the screen and scroll amount depends on. So, whatever, I am getting, may not be same for you.

Also, all these four combination works with a number prefix and by now you may have guessed what it does for you.

Now, on the same screen you may need to move around anywhere While in this case vi helps a little bit.
While you always have the option to use <number>G combination, vi also helps with the following three commands,

**H moves the cursor to the top line of the screen.
**M moves the cursor to the middle line of the screen.
**L moves the cursor to the last line of the screen.

Now pretty uncommon scenario but still you may need to scroll just a single line of the file. Following are two commands that you can use,

**Ctrl+y - forward scrolls one line
**Ctrl+e - backward scrolls one line

These were pretty common basic cursor movement commands. Get a good hold of them. Because, if you are on UNIX/Linux, pretty common that you'll be using vi editor and these commands will save you a lot of time.

That was all the basic commands used for cursor movement and you need to practice for a while and you will make it to a level where you can use these commands efficiently to ease your work. In the next article, we'll start with insert mode.

<div class='
