---
title: "vi editor: Programmers' choice - Part 1"
date: "2016-07-14T12:29:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021054157)"
---

Before going any further, we would like to start with an editor.
**Why an Editor ?

Once you are in Unix/Linux, you will most likely deal with many files and you may need to write you own scripts to work with the OS.

If you are a programmer, you might want to find a good editor for your source codes.

I will be describing one of the editors available for you in any UNIX/Linux distribution.The editor is known as **vi editor and for many programmer this editor suffices their needs.

So, let's start with this editor.

**What's the hype about this one ?
Well, vi is a screen based text editor which runs on terminal. Let me tell you some features of this editor.

No need of installation. With all major UNIX/Linux/Mac OS distribution you get this one bundled.

Choosing this editor helps you write your source code in any UNIX/Linux distribution. Even if you are on Windows you can install this editor.

Less memory footprint.

Does exactly what you need.

If used properly this can help you with lot of typing.

Very much user friendly.

Well, that's all the features I like about **vi. If you know some, please comment and I will add to this list.

Apart from these, you can directly take views from real time users here -
[https://www.quora.com/Why-are-tools-like-Vim-and-Emacs-still-used-for-coding](https://www.quora.com/Why-are-tools-like-Vim-and-Emacs-still-used-for-coding)
[https://www.quora.com/Why-do-some-developers-prefer-terminal-style-editors-like-vi-Vim-and-Emacs-instead-of-GUI-based-editors-like-Sublime-Text](https://www.quora.com/Why-do-some-developers-prefer-terminal-style-editors-like-vi-Vim-and-Emacs-instead-of-GUI-based-editors-like-Sublime-Text)

Now you may wonder how to use this editor.
Well, it's pretty simple, just type vi in the terminal and it invokes the editor.

For Windows users, there are some more steps to go ahead.

Navigate to this page with any of your favourite browser, [http://vim.sourceforge.net/download.php](http://vim.sourceforge.net/download.php) 

Move to this section PC: MS-DOS and MS-Windows

Install the editor

Add the installation path to the environment PATH variable

Open command prompt/Poweshell and type **vim

That's it we are all set to use this beautiful editor.
** To keep things simple, we'll start with an empty directory.

```
palash@palash-pc:~$ cd
palash@palash-pc:~$ mkdir vi-tut
palash@palash-pc:~$ cd vi-tut

```

If you are new to linux and do not know any command, you may find this difficult. Let me tell you the commands I used and what they do.

**cd - Change directory. When we want to move to any directory, we use this command. When *cd is entered to the command line without any argument, it moves to the home directory.
So, in the first line, we've moved to the home directory and in the third line, we've moved to a directory named *vi-tut.

**mkdir - Make a directory. If you want to create a directory, use this command and it will create an empty directory with the name specified as argument. If it finds a directory already present, it will throw an error. If you get any error, provide another name.

This is in short about the commands we've used in this article. We'll dive deeper in directory structure, naming conventions etc. etc. later. For now, these two command suffices more than enough.

We started with our home directory and created a directory named **vi-tut. We are currently in this directory.
Let's start our journey. We will go through some very basic concepts of this editor and then will get our hands dirty.

### Modes of vi editor

vi editor comes with keyboard-only concept. What I mean by this is that unlike other editors(like Notepad, Notepad++, GEdit, emacs etc) vi editor does not come with a menu to deal with file related tasks. Instead, it uses different key combinations for this purpose.

So, you might be thinking, if it uses different key combinations, don't they show up in the file content ?

Yes, you are thinking in a right direction. To handle this scenario, vi editor works on two different modes. First mode is to issue commands to the editor and the other mode is for edit purpose. So using the keyboard only we can deal with file contents.

Let's dive deeper on these two  modes.

#### Command Mode

This mode talks to the editor instead of printing the entered characters into the main content. It is the default when you open a file in vi editor.

#### Insert Mode

This mode is for dealing with the content of the actual file. When we type anything on the keyboard, the same goes to the content of the file.

All these seem to be very confusing at first. So, let's start our journey in the system.

```
[palash@palash-pc vi-tut]$ vi

```

This is command to invoke vi editor. The simplest of all...
So, this will open the editor and should see something like the following

```
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~                                                                          VIM - Vi IMproved
~
~                                                                           version 7.4.1910
~                                                                       by Bram Moolenaar et al.
~                                                             Vim is open source and freely distributable
~
~                                                                    Help poor children in Uganda!
~                                                            type  :help iccf<Enter>       for information
~
~                                                            type  :q<Enter>               to exit
~                                                            type  :help<Enter>  or  <F1>  for on-line help
~                                                            type  :help version7<Enter>   for version info
~
~
~
~
~
~
~
~
~
~
~
~
~
~
0,0-1         All

```

**Note: The color combination my vary depending on your terminal.

So, what it says ?
It tells you that you have opened the vi editor and it is running the improved version of vi editor with some other information as well.

Simply type **ZZ (note the capitals) and it will exit the editor.

Next we will try vi editor with an argument,

```
[palash@palash-pc vi-tut]$ vi myText.txt

```

This will open the editor once again but you should see some difference this time. Following is what we are seeing,

```
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
"myText.txt" [New File]

```

So, this time you are seeing the editor with no banner and also the file name is mentioned in the bottom left corner.

Now, let's start typing some text on this file. To enter something in the file, we need to enter the *Insert mode. To do so, type **i. And you are in *Insert mode.

```
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
-- INSERT --                                                                                                                                          0,1           All

```

Note the -- INSERT -- description on the bottom left corner and the Line Number, Character Number information in the bottom right corner.
Well, for some version of vi editor the *-- INSERT -- banner does not show. In that case, you might be in a confused state. To overcome this confusion, hit Esc key twice which will definitely change to command mode.

Now type something in the file and hit Esc key once or twice. It will go back to Command Mode. This is the following what I've written in the file and went back to Command Mode

```
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
"myText.txt" 3L, 129C

```

Now the text file "myText.txt" contains 3 Lines and 129 characters and all these information are available in the bottom of the editor. Now exit the editor. Following are the commands you can use to exit the editor.
**ZZ - Saves the content displayed on the screen into the file and exits the editor
**:wq - Saves the content displayed on the screen into the file and exits the editor.
**:w - Saves the content displayed on the screen into the file but does not exit the editor
**:q - Does not save the file and exit the editor. This command won't let you exit the editor if the content has changed or the file is in read only mode.
**:q! - Does not save the changes and exits the editor.

Now, you might have already experienced the following, while typing **ZZ into the command mode, nothing was written on the screen. But when you have entered **:w or **:wq or **:q or **:q! the characters entered are also displayed in the bottom left corner of the editor. If you have not noticed this, let me show you once more,

```
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog
A quick brown fox jumps over the lazy dog.
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
:wq

```

So, you see in the bottom left corner, **:wq is also printed.

This is also known as command line mode (command mode where you can type commands).

Once you are done, you can open the file in read only mode using the following command,

```
[palash@palash-pc vi-tut]$ vi -R myText.txt

```

This command will open the file in read only mode (note -R in the argument). So, you cannot change anything in this file. I will leave the experimental steps on the reader.

Or, you can use the following commands also to view the file content.

```
[palash@palash-pc vi-tut]$ more myText.txt
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog
A quick brown fox jumps over the lazy dog.

```

Now I want you to do one thing, create a new text file with name "quick-brown-fox.txt" and type this text "A quick brown fox jumps over the lazy dog." 50 times and save the content.
While saving the content, use different commands like **ZZ, :wq, :w, :q one at a time.
To create a new file with the name, you should type in the terminal,

```
vi quick-brown-fox.txt

```

We'll use this file in our next article on vi editor. One humble request to you, please do not copy paste the content, instead write each line. I know, it will be difficult but it's worth the value. It will not only improve skill on vi editor, it will also improve your typing skill. So, no cheating please. You are also encouraged to try some other things or key combination in command mode.

### Quick Recapitulation

From my personal experience, I can tell you that this may be one of the confusing articles you've ever read. So, give it some time, try to grasp the basics, take break, complete the task I've suggested, stay focused and have the appetite to learn something new.
To help you with this article, let me summarize what we tried here -

Tried to get the basic commands for directory handling with ***cd and ***mkdir

Knew about the very basic yet powerful editor in UNIX/Linux

Knew about the installation procedure in Windows OS as well

Knew about different modes of vi editor

Tried the editor with or without arguments

Knew how to appear in different modes of vi editor

Tried to save the content with different commands

Tried to read the file content using vi read only mode

Tried to use another command to read the content of the file

That's all for today. We'll next look into some more commands of vi editor which will ease our lives.

<div class='
