---
title: "Python Installation"
date: "2016-01-04T13:27:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021055912)"
---

You might have covered the introduction and may be now in a thought of writing programs using python.

Well, you can write your program right away. No issues. But you cannot actually run the program.

**So, what's the use of writing a program that actually does nothing ?

Well, don't worry, you will run programs written in python. Only thing you need to have is python interpreter.

**So, how to get that ?

Well, you need to install python on your system. Next I am going to detail this step for each type of OS.

**For Mac OS X

Good news for Mac OS X users, python actually comes pre-installed on your system. So, no need to download or install anything. Simply open your terminal and type **python, you will find that python is already working.

**For Unix

Most unix distributions also come with pre-installed python. So, most likely you don't have to download or install python on your system

You can verify if your system comes with python or not. For details, read the **Verification section.

If you don't see python installed on your system, you can check this article, [[Tutor] how to install python on unix machine if you don't have root](https://mail.python.org/pipermail/tutor/2002-March/012903.html)

**For Windows

Go to python website, [https://www.python.org](https://www.python.org/)

Go to Download tab and download version 2.7.x. **Do not download python 3.

It will download the installer.

Once download is complete, click the installer which starts the installation.

You will be asked if you want to install python for all users or for your user only. Choose as applicable for you.

Click next. It asks for installation directory. You can use the default directory or can change the location.

Click next

It will start the installation. Wait for some time to complete this step. You may get Security warning from Windows. Click Yes.

Once the installation is complete, click finish.

**For Android

If you want to learn python on the go and write some useful stuffs on your Android device and use it later, well python has that capability too. You don't need your laptop or desktop to run python applications. Let me guide you,

Go to Play Store

Search for QPython

Install the application

Now you can use the console, write programs and run them on your android device as well

Let me take you to this application a little bit.

![image](https://1.bp.blogspot.com/-BcuwLiPTnDc/VvLFDCeTZPI/AAAAAAAABFE/9pYKB273VJQPh7PjPKTPfVWHctS-HLjXg/s1600/Screenshot_20160323-215834.png)

This is the first screen, slide right to the next window,

![image](https://1.bp.blogspot.com/-Id6iEMWnmhY/VvLFSw3I0SI/AAAAAAAABFI/BA_8leFDRDkD5O7liKpuxAHdlauUeirGQ/s1600/Screenshot_20160323-215841.png)

You can select what you want to do. I am choosing console and let's see how this looks,

![image](https://1.bp.blogspot.com/-J6911WZA_Ks/VvLFjC8kqGI/AAAAAAAABFM/NKC9LdgUHfMMyJRbuc_stMkbTX2OMLEgQ/s1600/Screenshot_20160323-215848.png)

You can see that this screen looks almost the same as a standard PC console. Type some python commands on it and the console returns the result.

Now, let's look at the editor window.

![image](https://1.bp.blogspot.com/-0X_cq-CO7ec/VvLGMAUNNDI/AAAAAAAABFU/eaF0NYIuZx07WzxVb2GgCizs-vrlfSYTA/s1600/Screenshot_20160323-215903.png)

In this window, you can write your own python scripts, can save it, run it and other options you can explore.

In programs, you can view all the programs,

![image](https://4.bp.blogspot.com/-I9kws7EBOMI/VvLGp2W2BnI/AAAAAAAABFc/QqDVtqoY-PEFECMPTq6BmJYYERbjRBRug/s1600/Screenshot_20160323-215924.png)

Tap on a program and it will show you the available commands for the file like run, edit, rename etc.

The qedit4web.py is a beautiful program which allows you to connect to QPython over wi-fi to a PC. Check it out.

Cool, right ?

How are you planning to use it ?

**Verification

Once the installation is complete, open a terminal and type the following,

```
python
```

You should see similar to the following,

```
Python 2.7.11 (v2.7.11:6d1b6a68f775, Dec  5 2015, 20:32:19) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Congratulation, you have successfully installed python and it is ready to use.

Now let's do some simple calculations and greet people using python,

```
>>> 2+3
5
>>> 4*5
20
>>> 2/5
0
>>> 2.5/5
0.5
>>> 5+8
13
>>> print "hello"
hello
>>>exit()
```

Cool. right ?

**But what about writing some programs ?

Well, we'll discuss that in our next articles. But for this time, we'll write short program that we can run on python.

Let's write this short program which greets the world,

```
print "Hello World"
```

Open a text editor, write this line and save it as hello_world.py

Now open terminal and change directory to the directory where you have saved the file and run the following command,

```
D:\Palash\python\trunk\src>python hello_world.py
Hello World

D:\Palash\python\trunk\src>
```

And you can see that your program greets the world.

**SAME OLD HELLO WORLD.

Yeah, this way of greeting the whole world is very much common in use, how about actually greeting the user who is running the program ????

**Interesting !!!

**
OK, let's write the following programming which asks the user to input his name...

```
person = raw_input('Enter your name: ')
print 'Hello', person
```

Now write these lines in a file and save it as hello_user.py and run the file using python command,

```
D:\Palash\python\trunk\src>python hello_user.py
Enter your name: Palash Kanti Kundu
Hello Palash Kanti Kundu

D:\Palash\python\trunk\src>
```

When you will run the program, the system will ask you to enter your name and wait until you enter your name. Once you have entered your name and hit return, you will see the output which actually greets you.

Let's run this program again,

```
D:\Palash\python\trunk\src>python hello_user.py
Enter your name: Suman Kundu
Hello Suman Kundu

D:\Palash\python\trunk\src>
```

Now you see, you get a different output based on the user input. Now let's run this program for some more time,

```
D:\Palash\python\trunk\src>python hello_user.py
Enter your name: Totan das Kundu
Hello Totan das Kundu
```

```

```

```
D:\Palash\python\trunk\src>python hello_user.py
Enter your name: Subhra Datta Roy
Hello Subhra Datta Roy
```

D:\Palash\python\trunk\src>python hello_user.py Enter your name: Radhika Kittur Hello Radhika Kittur  D:\Palash\python\trunk\src>python hello_user.py Enter your name: Amrita Pattanaik Hello Amrita Pattanaik  D:\Palash\python\trunk\src>python hello_user.py Enter your name: Annu Singh Hello Annu Singh  D:\Palash\python\trunk\src>python hello_user.py Enter your name: ANY_NAME Hello ANY_NAME

That's it. You have successfully installed python, performed some mathematical calculations, greeted the world, greeted the user who is using the program.

Well, in our next sections, we'll be discussing the way you write python programs and its different syntax and all.

Have fun till then...

### Source Code Distribution

The source codes associated with this articles can be downloaded from github,

[hello_world.py](https://github.com/Palash90/python/blob/master/src/hello_world.py)

[hello_user.py](https://github.com/Palash90/python/blob/master/src/hello_user.py)

[Prev](http://python.palash90.in/2016/01/introduction-to-python.html)     [Next](http://python.palash90.in/2016/01/python-shell.html)

<div class='
