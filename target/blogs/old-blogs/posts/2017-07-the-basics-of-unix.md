---
title: "The basics of UNIX"
date: "2017-07-16T12:28:00+05:30"
recovered_from: "web.archive.org (snapshot 20190917002822)"
---

With our previous post, we knew that UNIX is an operating system which was developed in and Bells Labs back in 1970 and after that many versions of UNIX have become available and some UNIX Like Operating Systems like Linux has came into picture. In the time line Linux has also gained popularity and most of the people today use Linux whether they know it or not.

We have taken look on the big picture. Now, we'll be looking into the basics of the Operating System.

UNIX/Linux has the following programs at its bare minimum,

#### Kernel

The programs that deals with the hardware directly. This is the main building block of the OS. These basically allocates time and memory of other programs. Also deals with low level file structure and communication related works.

#### Shell

Shell can be thought of a bridge between the Kernel and the user. It is a Comman Line Interpreter (CLI). When user types in any command in the Shell, the shell finds the exact program that deals with the command and invokes the kernel.

There are many versions of shell available. Each comes with its own features.
To name a few different Unix/Linux shells,

B Shell

C Shell

K Shell

Bash Shell

tcsh

Zsh

Don't worry, if you don't know which shell you are using, if yo are defining anything by your own, you will be provided with the system default. Apart from the features a shell provides, it does the basic thing, interpret the command you key in. So, you are safe with any shell. Once you know the basic commands and shell programming, you wil find the best suit for you.

#### File and Processes

Well, in UNIX everything can be categorized in either of these types, File/Process.
A file is a collection of data. As I mention data, it may be anything, a document, a C program, a media file, an executable. Files are arranged in directories.

Each file has a name associated with it which works as an unique identifier for the file within a directory. Similar goes for a directory, within a single parent directory, two sub-directories cannot have same name.

Now coming to process, a process is a program that is being executed in the hardware. A process has also a unique identifier for it and is known as PID (Proces Identifier). We can perform many operations by our own with this PID.

#### Directories

**As** we've known** that files are arranged into directories and in the similar fashion, directories can be again arranged into directories.** The hierarchy** starts usually with root directory** and is usually marked with front slash** (/). If we think this a graph, then directories are the node where files are the leaf nodes of the graph.**
Let's look at a directory hierarchy,
![image](https://1.bp.blogspot.com/-bwlQd-hl2pU/VsyXkHEFKGI/AAAAAAAABAA/Qidwoj5V64g/s1600/Untitled.png)
In this hierarchy, we can see that **root directory has two directories *bin and *home(There are many more directories present in the root. This is just for demonstration purpose. Don't get confused.). So, *bin and *home are sibling. In *home directory, we see another two directories *palash and *Anyone. Similarly, in *palash we've *git and within that we've *me and within *me we find a file **pom.xml. So, except **pom.xml all others are directories and *bin & *home are sibling while *palash & *Anyone are sibling.

Now, let's look into the running processes. To see all the running processes, issue the following command in the shell, To open a shell press **Ctrl+Alt+T. This is the default key combination for opening a terminal.

```
ps -ef

```

You will see the output similar to the following,

```
UID         PID   PPID  C STIME TTY          TIME CMD
root          1      0  0 Feb23 ?        00:00:19 /sbin/init
root          2      0  0 Feb23 ?        00:00:00 [kthreadd]
root          3      2  0 Feb23 ?        00:00:04 [ksoftirqd/0]
root          4      2  0 Feb23 ?        00:00:00 [kworker/0:0]
root          5      2  0 Feb23 ?        00:00:00 [kworker/0:0H]
root          7      2  0 Feb23 ?        00:00:32 [rcu_sched]
root          8      2  0 Feb23 ?        00:00:00 [rcu_bh]
root          9      2  0 Feb23 ?        00:00:28 [rcuos/0]
root         10      2  0 Feb23 ?        00:00:00 [rcuob/0]
root         11      2  0 Feb23 ?        00:00:01 [migration/0]
root         12      2  0 Feb23 ?        00:00:00 [watchdog/0]
root         13      2  0 Feb23 ?        00:00:00 [watchdog/1]
root         14      2  0 Feb23 ?        00:00:00 [migration/1]
root         15      2  0 Feb23 ?        00:00:03 [ksoftirqd/1]
root         17      2  0 Feb23 ?        00:00:00 [kworker/1:0H]
root         18      2  0 Feb23 ?        00:00:06 [rcuos/1]
root         19      2  0 Feb23 ?        00:00:00 [rcuob/1]
root         20      2  0 Feb23 ?        00:00:00 [khelper]
root         21      2  0 Feb23 ?        00:00:00 [kdevtmpfs]
root         22      2  0 Feb23 ?        00:00:00 [netns]
root         23      2  0 Feb23 ?        00:00:00 [perf]
root         24      2  0 Feb23 ?        00:00:00 [khungtaskd]
root         25      2  0 Feb23 ?        00:00:00 [writeback]
root         26      2  0 Feb23 ?        00:00:00 [ksmd]
root         27      2  0 Feb23 ?        00:00:19 [khugepaged]
root         28      2  0 Feb23 ?        00:00:00 [crypto]
root         29      2  0 Feb23 ?        00:00:00 [kintegrityd]
root         30      2  0 Feb23 ?        00:00:00 [bioset]
root         31      2  0 Feb23 ?        00:00:00 [kblockd]
root         32      2  0 Feb23 ?        00:00:00 [ata_sff]
root         33      2  0 Feb23 ?        00:00:00 [md]
root         34      2  0 Feb23 ?        00:00:00 [devfreq_wq]
root         36      2  0 Feb23 ?        00:00:07 [kworker/1:1]
root         38      2  0 Feb23 ?        00:00:00 [kswapd0]
root         39      2  0 Feb23 ?        00:00:00 [fsnotify_mark]
root         40      2  0 Feb23 ?        00:00:00 [ecryptfs-kthrea]
root         52      2  0 Feb23 ?        00:00:00 [kthrotld]
root         53      2  0 Feb23 ?        00:00:00 [acpi_thermal_pm]
root         54      2  0 Feb23 ?        00:00:00 [scsi_eh_0]
root         55      2  0 Feb23 ?        00:00:00 [scsi_tmf_0]
root         56      2  0 Feb23 ?        00:00:00 [scsi_eh_1]
root         57      2  0 Feb23 ?        00:00:00 [scsi_tmf_1]
root         59      2  0 Feb23 ?        00:00:29 [kworker/0:1]
root         63      2  0 Feb23 ?        00:00:00 [ipv6_addrconf]
root         84      2  0 Feb23 ?        00:00:00 [deferwq]
root         85      2  0 Feb23 ?        00:00:00 [charger_manager]
root        139      2  0 Feb23 ?        00:00:00 [mpt_poll_0]
root        140      2  0 Feb23 ?        00:00:00 [mpt/0]
root        141      2  0 Feb23 ?        00:00:00 [kpsmoused]
root        142      2  0 Feb23 ?        00:00:00 [kworker/1:2]
root        143      2  0 Feb23 ?        00:00:00 [scsi_eh_2]
root        144      2  0 Feb23 ?        00:00:00 [scsi_tmf_2]
root        146      2  0 Feb23 ?        00:00:00 [scsi_eh_3]
root        147      2  0 Feb23 ?        00:00:00 [scsi_tmf_3]
root        148      2  0 Feb23 ?        00:00:00 [scsi_eh_4]
root        149      2  0 Feb23 ?        00:00:00 [scsi_tmf_4]
root        150      2  0 Feb23 ?        00:00:00 [scsi_eh_5]

```

Well, these are some processes running in my system. This list may vary from system to system. Also the output may vary depending on the system you are working.
For example, this output was taken from Ubuntu 14.04 while in FreeBSD you will get some different output. But one thing will be common in all the output, PID. This shows the system generated identifier for the process.

That's all what I can tell you for now about the very much basics of UNIX/Linux.

Happy learning...

<div class='
