---
title: "Dealing with long running commands"
date: "2017-07-16T12:23:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045043)"
---

Working with **bc and other commands, we gained confidence and it is now quite easy for us to run commands in the shell. We are typing the commands and shell is working for us (especially those boring maths !!!), pretty nice, isn't it ?
But this may not be the case all the time, sometimes a few commands take a lot of time to run. Now, we will take a look at what we can do in those cases. So, let's first run a long running command in shell. We will use the **yes command for this purpose. **yes is a command that prints  **y continuously on the screen. Let's see how -

```
totan@Home-Computer ~ $ yes
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y
y

```

and if you run this command (but choose not to run this command until you complete reading this article), you can not run any other command. You may think that you are seeing the same screen, nothing is changing. So I will now write a program of one line and make it clear to you, that we are indeed running something that will run infinitely. If you read all the **bc related articles, you may understand this program -

```
totan@Home-Computer~$sh
totan@Home-Computer~$while true; do date; sleep 1; done;
```

At first I changed my shell to **sh and ran a small one-line program. Now let's see what its output is -

```
Tue Jul 25 09:27:17 IST 2017
Tue Jul 25 09:27:18 IST 2017
Tue Jul 25 09:27:19 IST 2017
Tue Jul 25 09:27:20 IST 2017
Tue Jul 25 09:27:21 IST 2017
Tue Jul 25 09:27:22 IST 2017
Tue Jul 25 09:27:23 IST 2017
Tue Jul 25 09:27:24 IST 2017
Tue Jul 25 09:27:25 IST 2017
Tue Jul 25 09:27:26 IST 2017
Tue Jul 25 09:27:27 IST 2017
Tue Jul 25 09:27:28 IST 2017
Tue Jul 25 09:27:29 IST 2017
Tue Jul 25 09:27:30 IST 2017
Tue Jul 25 09:27:31 IST 2017
Tue Jul 25 09:27:32 IST 2017
Tue Jul 25 09:27:33 IST 2017
Tue Jul 25 09:27:34 IST 2017
Tue Jul 25 09:27:35 IST 2017
Tue Jul 25 09:27:36 IST 2017
Tue Jul 25 09:27:37 IST 2017
Tue Jul 25 09:27:38 IST 2017
Tue Jul 25 09:27:39 IST 2017
Tue Jul 25 09:27:40 IST 2017
Tue Jul 25 09:27:41 IST 2017
Tue Jul 25 09:27:42 IST 2017
Tue Jul 25 09:27:43 IST 2017
Tue Jul 25 09:27:44 IST 2017
Tue Jul 25 09:27:45 IST 2017
Tue Jul 25 09:27:46 IST 2017
Tue Jul 25 09:27:47 IST 2017
Tue Jul 25 09:27:48 IST 2017
Tue Jul 25 09:27:49 IST 2017
Tue Jul 25 09:27:50 IST 2017
Tue Jul 25 09:27:51 IST 2017
Tue Jul 25 09:27:52 IST 2017
Tue Jul 25 09:27:53 IST 2017
Tue Jul 25 09:27:54 IST 2017
Tue Jul 25 09:27:55 IST 2017
Tue Jul 25 09:27:56 IST 2017
Tue Jul 25 09:27:57 IST 2017
Tue Jul 25 09:27:58 IST 2017
Tue Jul 25 09:27:59 IST 2017
Tue Jul 25 09:28:00 IST 2017
Tue Jul 25 09:28:01 IST 2017
Tue Jul 25 09:28:02 IST 2017
Tue Jul 25 09:28:03 IST 2017
Tue Jul 25 09:28:04 IST 2017
Tue Jul 25 09:28:05 IST 2017
Tue Jul 25 09:28:06 IST 2017
Tue Jul 25 09:28:07 IST 2017
Tue Jul 25 09:28:08 IST 2017
Tue Jul 25 09:28:09 IST 2017
Tue Jul 25 09:28:10 IST 2017
Tue Jul 25 09:28:11 IST 2017
Tue Jul 25 09:28:12 IST 2017
Tue Jul 25 09:28:13 IST 2017
Tue Jul 25 09:28:14 IST 2017
Tue Jul 25 09:28:15 IST 2017
Tue Jul 25 09:28:16 IST 2017
Tue Jul 25 09:28:17 IST 2017
Tue Jul 25 09:28:18 IST 2017
Tue Jul 25 09:28:19 IST 2017
Tue Jul 25 09:28:20 IST 2017
Tue Jul 25 09:28:21 IST 2017
Tue Jul 25 09:28:22 IST 2017
Tue Jul 25 09:28:23 IST 2017
Tue Jul 25 09:28:24 IST 2017
Tue Jul 25 09:28:25 IST 2017
Tue Jul 25 09:28:26 IST 2017
Tue Jul 25 09:28:27 IST 2017
Tue Jul 25 09:28:28 IST 2017
Tue Jul 25 09:28:29 IST 2017
Tue Jul 25 09:28:30 IST 2017
Tue Jul 25 09:28:31 IST 2017
Tue Jul 25 09:28:32 IST 2017
Tue Jul 25 09:28:33 IST 2017

```

You see, every second it is printing time on screen, if you do not stop it, then it will continue running until you close the terminal or turn off the computer. So it is very important for us to know some commands or shortcuts right now, with which we can work with these long running/infinite commands.
The first shortcut we will learn is **Ctrl-s , when this key combination is pressed, the running commands are paused, you may feel that the program has stopped running but you do not see the prompt and if you want to see what happens, then press **Ctrl-q. Now you try these commands and combinations and try to understand what is going on. But be careful, since the terminal or shell itself is a command, **Ctrl-s can pause your terminal program in cases, so that you may feel that the command line has stopped working or hung up and it is not accepting inputs. Press **Ctrl-q , to check whether the command line is actually paused or not.
Now this is time to know another powerful shortcut **Ctrl-c, this shortcut stops any running command and returns the shell prompt. Let's see - 

```
totan@Home-Computer ~ $ while true; do date; sleep 1; done
Tue Jul 25 09:49:03 IST 2017
Tue Jul 25 09:49:04 IST 2017
Tue Jul 25 09:49:05 IST 2017
Tue Jul 25 09:49:06 IST 2017
^C
totan@Home-Computer ~ $

```

The last line in the output is **^C, in the command line **Ctrl is shown as **^ So ^C in the shell means that **Ctrl-c input has been provided, which is actually an interrupt (later we will have a deeper analysis on this). In simple words shell has been instructed to stop the current command, so the shell will stop the current command and will return the prompt to the user. This way we can stop the current command if necessary.**So far we have worked with some very basic shell commands that will help us to work with the command line, we will now see how we can work easily with files in shell.

<div class='
