---
title: "What and why UNIX ?"
date: "2017-07-16T12:29:00+05:30"
recovered_from: "web.archive.org (snapshot 20190917003746)"
---

Welcome to the series devoted to the base platform of deployment.

In this article we'll look into the basics of computers and very less on the applications that run on any computer.

Many of us use different applications for different purposes like surfing the internet, playing games, watching movies, writing codes and for hell lot of things we use different applications. We use applications not only in our PCs but also in other devices like mobile phones, tablets, embedded systems, Printers, TVs, CD Players, Cars, ATMs, Micro-ovens and in many other devices.

You might be wondering how does a car have a relation to an application.

Well, it does. Wherever you see anything tagged with the word 'DIGITAL', application follows you. You may know it or may be the application is abstracted in a way that you cannot understand it. But the truth is that, when you use a digital device you are basically using an application.

People play Media Files using Windows Media Player or VLC Media Player, surf internet using Mozilla Firefox or Google Chrome. Many programmer like us use the following commands day by day,

```
gcc
g++
javac
java
python
rustc
./a.out

```

When asked, how does all these work, many can answer it while most cannot answer.

As a programmer, we should have a basic understanding of the system. If you are not a programmer also, you should know this for your own knowledge.

So, let's start with the discussion.

Whenever we issue any command or start running any application (like Windows Media Player) we see the work is being done.

Who does the work for you ?
Well the box sitting in front of you known as computer.

May be you have opened your box and have seen that there are some boards embedded into it with lots of pins exposed, some soldered silicon chips(looking more like the memory chip you have in your mobile), some chips are blue, some are green, some yellow and mostly black. You have seen a lot of wires plugged here and there. Something like the following,

<tbody> | ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/PDP-11-M7270.jpg/220px-PDP-11-M7270.jpg)
| Image Courtesy: [Wikipedia](https://en.wikipedia.org/wiki/Computer_hardware).
&lt;/tbody&gt;

Now, you think, how does all these wires and silicon chips understand what we want to do !

Puzzled and again screwed your box and went back to do something else.

Now, here is the answer.
The computer never understands what you want to do. Neither it understands, 'A, B, C, D, E'.

**Then ?
A computer only understands **0 and **1. Yes, you read it correctly, only **0 and **1. Or in a very simple way, if a switch is turned on or turned off.
**Then, how am I reading this article ?
**It is full of alphabets and not *0 & 1. Nowhere it is showing any switch as well.
Well, it is true that you are reading this article which is in English but still I'll say that computer does not understand English.

Well, no more confusion, let me tell you how does it do that.

There are certain applications built for that and those are the production of programmers. These programs are responsible for converting all the 0s and 1s to English and vice versa.

Now let me take you to a visual experience of the whole end to end process.![image](https://2.bp.blogspot.com/-upQrKCTtf_g/VsWaMeiVOkI/AAAAAAAAA_k/94JbyheabEc/s1600/Untitled.png)

When using any device, we just see it as a device and it takes our instruction and works for us. But in reality the process is not so simple, in fact this process of instructing your device encompasses many more things in it.

If you look into the diagram, you can understand that there are 3 components involved in it, Applications, Operating System and Hardware.

Let me go through each of them in detail,

**Application: The interface between the user and the computer. The user provides input to the application and the application takes it forward to the Operating System.

**Operating System: Operating System is nothing but a set of applications that can actually talk to the hardware directly. These applications are particularly known as **kernels which actually deal with the hardware.

**Hardware: The actual worker. The hardware deals with the instructions Operating System (to be precise, kernels) has given to it.

The hardware performs the instruction and sends the result back to the Operating System and it correspondingly sends back the output to the application.which actually renders the result for the user.

Well, this is a 10,000 ft view of what goes on in the computer. Actually there are many processes involved in it.

Now, you get a clear picture of how things are dealt in computer world.

Now, let me take the opportunity to answer the question we started with.

**What is UNIX ?
Well, UNIX is an Operating System built in 1970 at Bell Labs research Center by Ken Thompson, Dennis Ritchie and others.

Actually, from then it has evolved a lot much and many more Operating Systems have been developed which behaves like UNIX. One of them is Linux, written by Linus Torvalds in 1991. Linux is nowadays the ruling OS spread worldwide.

If you are interested, you can go through the following articles.

[Unix Impact](https://en.wikipedia.org/wiki/Unix#Impact)

[Unix Like](https://en.wikipedia.org/wiki/Unix-like)

[Linux](https://en.wikipedia.org/wiki/Linux)

[Android](https://en.wikipedia.org/wiki/Android_(operating_system))

[Unix](http://www.unix.org/what_is_unix.html)

Many operating systems are influenced by UNIX and we have many distributions of Unix available.

You may be wondering, why take the pain to learn about it ?
Let me clear your thoughts,

An OS by developers for developers

Better hardware management

Improved productivity

Gaining the knowledge on OS Model

Unix is run mostly on the deployment servers. Almost 99% servers are UNIX

All the major websites(including Google, Facebook, Amazon) today use UNIX for hosting

If you are a developer, most probably your application is also going to be hosted in Unix

A lion share of people deal with devices which run Android/iOS which are Unixy in nature

Unix-Like OS Linux is open source and has a great community support. So, after some days working with Linux you may also change your basic OS from Windows to Linux. This is what happened when I started my first Linux usage in my 5th Semester back in 2009.

Those were some of the points many people discuss. Now let me tell you my own story where a bit  of Unix knowledge helped our team.

Few years ago, we were working in a very critical memory savvy project and we had to fix all the memory leakage in the application. Our team really did a great job and we fixed all the areas where memory leak can occur. The application was built and deployed. The first deployment failed and the System Administrator blamed us that we did not fix anything and simply put the old code in deployment. We again checked every suspected area and deployed, the same thing happened and the same thing repeated four to five times.

This time we were there with the SA while deploying and we found that he was deploying 10 applications in a single go in the server and it was throwing error of insufficient buffer. We realized that only seeing a memory issue, he blamed our team, not even looking into the root cause of Deployment Failure.

Our team checked the system log line by line and found that it was not our application which was creating problem rather it was the procedure of deployment.

I still remember the day when the architecture team praised our team for finding out the real deployment issue.

If our team members could not find the Unix Administration issues, we would still be blamed for not fixing the memory leaks.

That day I realized that developers should also learn Unix at least to such a level that it helps you find out the systems errors and the System Administrators can not blame you without proper reason.

Now with all that, we should be looking into the distribution/version we should use for our learning purpose.

As mentioned above, Unix has many versions of it while some Unix-Like Operating Systems like Linux has different distributions.

So, the question arises which version to use and how to use it ?
Each distribution of Unix/Linux is meant for a specific reason and keeping specific goals in mind. But in all these, one thing is common that is the shell.
When we want to explore the basics of UNIX, we will mostly look into the details of Shell commands and different system configurations through shell.
In many cases people want to write some programs dealing with different areas of the Operating Systems like File System, Networking etc.

So, ideally for this series any Unix/Linux distribution will work fine.

But if you are very new to Unix/Linux, you can start with **Ubuntu 14.04. This OS has a faster response, better GUI, easier set-up and almost Windows Like GUI.

In my case, I will work with both **Ubuntu 14.04 and **FreeBSD 10.2. If you are using other distributions, chances are some commands need a small change. Which you can get help from the particular distribution community.

Now you may want to know how you can use either of these. I will try to provide some suggestions.

**Caution: Instructions below are dangerous to try without proper knowledge and you may loose all your data if done incorrectly. So, if you do not know how to format your PC, don't try these. Instead, ask someone for help.

### Download and Install

Choose any open source version of Unix/Linux distribution. Here is the link for [top 10 Distribuiton List](http://distrowatch.com/dwres.php?resource=major). Once you are done with the selection, download the distribution and burn the content to a DVD and boot your system with the DVD. As an alternative, if you BIOS supports booting from Pen Drive, you can also make your Pen Drive bootable and boot with it.

Once you are in, you may or may not install it. You can simply try the Distribution and on your choice can install it or simply get out of it without making any change to your system.

### Try inside a VM Box

If you are not in the mode of installing the whole system on your PC and remove everything else or you want to try different flavors of Unix/Linux, you may use this option. Only drawback of this is, it requires high Hard Disk space.

Same goes here for choosing the distribution and download.
Now, you need to work with a VM Box. You can use VMWare Player or Oracle VM Virtual Box for this purpose.

<u>**Please do not use these tools for commercial purposes. They are meant to be used personally.

Here are links,

[https://www.virtualbox.org/](https://www.virtualbox.org/)

[https://www.vmware.com/products/player](https://www.vmware.com/products/player)

You can try both and according to your choice, you can keep either one or both. They both work together without hampering your base system.

Now install the OS inside the VM Box with downloaded image of UNIX/Linux Distribution.

Now, that's all about UNIX/Linux.

You can now understand what UNIX is and what is its use.

You surely have understood that **Linux is not UNIX.

You may have got an idea how many devices are **Unixy in nature.

You also know what are the distributions available of Unix/Linux.

You may have chosen your distribution.

You know how to use Unix/Linux. Another reminder, if you are not aware of system format, please ask for help from your friends or any computer vendor.

May be you are planning for Virtual Box installation of the Unix/Linux Distribution.

Hope that, you enjoyed this article...

<div class='
