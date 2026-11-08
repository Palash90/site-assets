---
title: "Terminal, Console, Shell, Kernel, Commands - Different parts of a Computer"
date: "2017-07-16T12:27:00+05:30"
recovered_from: "web.archive.org (snapshot 20190917002757)"
---

We somewhat interchange the terms Console, Terminal, Shell. Also, in contexts, we come across the word Kernel. So, this creates a confusion while first starting with Unix-like Operating systems. In this article, we will start digging into it to understand, what they mean.

To really know the meaning of all these, you have to go back to the era of computers sized as a room. Otherwise, it’s not possible for you to get it. As technology evolved and advanced day by day, these all came into a single place. Because, in today’s world all these are in a single box and you cannot really differentiate much. But remember, all these work independently still today, all these are still different parts in today’s world as well. Just, the difference is, they all live in the same place as opposed to the earlier when they used to be one single component of their own.
Enough intro, let’s start differentiating.
First I will go through the **Kernel and **Shell.
You must have seen an walnut, which looks something like this,
![image](https://1.bp.blogspot.com/-_utESslLgSY/WWA4xjSaKoI/AAAAAAAAG0s/zl-iCaqXUio3zmiubSdOjaJrDqXQGtAZgCLcBGAs/s1600/Walnut2-700x525.jpg)

[Image Courtesy: [11 Incredible Benefits of Walnuts Nutrition](https://www.organicfacts.net/health-benefits/seed-and-nut/health-benefits-of-walnuts.html)]

What is the main part of it ?
You might throw me something like, “Obviously, the inner part of it. Come on, is it a question to ask at all ?”
You know, what it is called, right ?
Yes, they are known as **Kernel and what is the part called which wraps the kernel ?
You might connect the dots right now. Yes, it is known as **Shell.

Yes, there is the connection of these two. So, the outer cover of Kernel is Shell, where Kernel is the important core part of the Operating System which directly talks to the hardware. So, shell is what connects the kernel to the user. It all dates back to 1960’s when computers used to do some specific tasks and each computer was different in their own flovors and working on one computer was not enough to certify you as a computer genius. Because, to work on different computers, you had to know them differently as well.

But in 1970’s in Bell Labs some really genius people ([Ken Thompson - Wikipedia](https://en.wikipedia.org/wiki/Ken_Thompson), [Dennis Ritchie - Wikipedia](https://en.wikipedia.org/wiki/Dennis_Ritchie)) mostly noted for their contribution) started thinking of a solution of this problem and they thought of a process of minimising the stuffs and distinguished each part of a computer. They found that, only a smaller part of the computer systems needs to be changed to work differently for different computers and all other stuffs can be ported to any other computers without much hassle. That’s where the concept of kernel comes in. Kernel is actually that part which needs to be different for each computer architecture and other stuffs can be same for each of them. Shell is one of those programs which can be similar for all kernels running on different computers.

So, that was somewhat on Shell and Kernel. Now, let’s look into the console and terminal part.
After devising the great mechanism for unifying all the computers under a single operating system also, did not resolve every problem. It was just a part of it. You remember, there is still the risk of having big computers, known as mainframes., which looked something like the following -

![image](https://2.bp.blogspot.com/-E1RjPViWUIQ/WWA4_5OZOgI/AAAAAAAAG0w/zdBjRdZyc0QoFrXYBRDgonLl8BqAqO4dwCLcBGAs/s1600/kd14.jpg)

[Image Courtesy: [Origins and History of Unix, 1969-1995](http://homepage.cs.uri.edu/~thenry/resources/unix_art/ch02s01.html)]

The above image is indeed a great picture of Ken Thompson and Dennis Ritchie working on mainframes.

This was how it looked at that time. So, you can understand that, it was fairly impossible to accommodate this really huge machine in each place. So, it was a practice to have a system in a central place and that system was communicated to multiple other smaller systems with almost no processing capacity. These smaller machines are those where users used to log in and connect to the main system (also known as Mainframe). This smaller machine was known as **Terminal, which looked like following -

![image](https://1.bp.blogspot.com/-mIptVLk4a7M/WWA5R0OnxEI/AAAAAAAAG00/2r6eIvngdY08CwqcnomEPmSt7UKcIlyTQCLcBGAs/s1600/DEC_VT100_terminal.jpg)

[Image Courtesy: [Computer terminal - Wikipedia](https://en.wikipedia.org/wiki/Computer_terminal)]

The terminals used to connect to the Mainframes using telephone connections. Maybe, you have heard the term Dial Up Connection.

Now, let’s talk about the console. Console is a special type terminal usually directly attached to the Mainframe for administration purposes. You may understand why the architecture is like that. If for any reason, the Mainframe fails to start up or may be the network connections cannot be established by the terminals to the mainframes, there should be some way to control the machine or to find out the error. That’s where the console comes to picture.

![image](https://3.bp.blogspot.com/-9COwW2wKjck/WWA5XGsVb9I/AAAAAAAAG04/lw06FJmyE182X3rh-dDberQXIUyWBKpxACLcBGAs/s1600/IBM_1620_Model_1.jpg)

[Image Courtesy: [Computer console - Wikipedia](https://en.wikipedia.org/wiki/Computer_console)]

Nowadays, all these pieces of different hardware are fused into a single piece and hundred times smaller than the original. So, basically, you can see them co-exist. So, basically we interchange them nowadays. But what you see in modern computers, is what should be correctly known as **Terminal Emulator.

So, these are the basic differences of a computer system in a lower level. Now, let’s step forward and look at higher level. An application is a program based on shell. Any program does not directly interact with Kernel, it usually talks to shell instead and shell executes required kernel instructions. Here comes the concept of Command Line Interface. So, basically, it means that, as a user, you will provide the computer some text based instructions (known as commands) via terminal. And it will connect to the mainframe via shell. In the mainframe, we have shell which talks to the kernel and kernel executes necessary hardware operations and in turn will return text based output on the terminal via shell. These commands are nothing but application programs. When commands are sent to shells, it will decide which application to run for the particular program. In turn applications also use shell to execute the required kernel instruction. It may be surprising to know that all the graphical user interface (e.g. Windows, Mac OS, Linux Graphical Desktop etc.) actually works in the similar model and connect to shell. We'll see how it is done later in this series.

Now, we can draw a big picture -
![image](https://2.bp.blogspot.com/-p4DWcWZeB2I/WV-1QKCoBfI/AAAAAAAAG0Q/_i9E6xAtJpMaTkhFIYr_YeLHWMqSv5SfgCLcBGAs/s1600/Untitled.png)

<div class='
