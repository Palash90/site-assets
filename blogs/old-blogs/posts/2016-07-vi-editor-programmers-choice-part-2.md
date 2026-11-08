---
title: "vi editor: Programmers' choice - Part 2"
date: "2016-07-14T12:28:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021044958)"
---

So we are good to deal with file opening, inserting some text in the file and saving it in the vi editor. In this post, we'll be looking into some more details like how to move the cursor to some place, how to copy and paste some words or a whole line etc.

So, in our earlier discussions, we have created a file with 50 lines of text and we will be using the same file to demonstrate some other commands of vi editor.

Open the file using vi command. Remember all of this post works in **Command Mode and not in Insert mode. So, be careful about what you are doing or you will end up by messing up everything.

Now going any further, let's print the line numbers in the file. To do that, simply write in the command mode,

```
:set number

```

![image](https://3.bp.blogspot.com/-KZXyRa_pjk4/V4erQCuepFI/AAAAAAAACtI/USloUQXs0YgDxhCo9sPkWKgINcUsGWt7gCLcB/s1600/Untitled.png)

So, in the bottom left corner, you can see, I've entered the command to show the line numbers. This is pretty useful when dealing with long files. Let's see what happens when we hit enter after typing the command,

![image](https://1.bp.blogspot.com/-mIqNuzaUKok/V4esqshuVKI/AAAAAAAACtY/DYUMHf2dN0c99R3OAq7aLT9u6ia7mzcAQCLcB/s1600/Untitled.png)

Now you can see that in each line the corresponding line numbers are shown. Again, note that the color combination in your system may vary from mine based on the version of vi and the terminal profile.

Now, let's start with moving the cursor in any direction. First move to command mode and try the following.
Following are the required keystrokes to move the cursor.
**h - moves the cursor to one character left
**j - moves the cursor to the lower line
**k - moves the cursor to the upper line
**l - moves the cursor to one character right

So if you press the key combination in succession, you will return back to the same character you started with **h k l j. Play with this combination couple of times and come back.

Now let's try these commands as prefix,
move the cursor to the middle of a line and try the following commands
**5h 5j 5k 5l
or
**3h 3k 3l 3j
or,
**10h 10j 10k 10l
or
<number>h <number>j <number>k <number>l
This time you will see that instead of moving a character, the cursor is moving the specified number of characters. For example **5h will move the cursors 5 characters left and if there are not enough characters, it will move to the first character. Similar goes with **j k and **l.

Now, this all about moving by characters, we can also move by words. Here is how we do that.
**w - Moves the cursor forward one word
**b - Moves the cursor backwards one word. If the cursor is in between of a word, it moves to the starting of the word.
**e - Moves the cursor to the end of the word.

You can also add number prefix with this commands to move the cursor to a specified amount of words. So, **5w will move the cursor to 5 words forward and so on.

Now, if you have noticed that vi editor is quite case sensitive (remember we used **ZZ to save our file and not <strike>zz&lt;/strike&gt;). So, if you try **W B E instead of w b e, you will notice that, these commands will ignore punctuation. Here also you can add number prefix to move forward or backward more than one word in a line.
Now these are the shortcuts for moving through characters and words. Next we'll check out shortcuts for moving through lines.

**$ - will move the cursor to the end of the line
**0 - will move the cursor to the beginning of the line

With lots of moving around, you may forget where you  started or where you are. So, you need the information to be shown. You have three options to do that,

Already discussed about **:set number command, you can use this command

You can use **Ctrl + g combination to get the information in the bottom left corner

Use **:set ruler command to show the current position of cursor in the bottom right corner

Now we'll check where the file is really long and we need to jump around from here to there. So, we'll be needing larger file. So, I would request the reader to again work on their typing skill and write the same content to make it 150 lines of same text, "A quick brown fox jumps over the lazy dog.".

Too much of typing, right ?
Yes I know. But it will help you in future. Even I am typing all these. So why can't you ?

Take a break, have some coffee, may be walk around for some time and come back once fresh. Type another 100 lines and come back. In the mean time let me write on the next part of this series.

Happy typing...

<div class='
