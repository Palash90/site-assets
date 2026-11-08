---
title: "Creating Files and Directories"
date: "2017-07-16T12:18:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045054)"
---

So far, we have gathered different type of information about file system. Many times it may be difficult to understand how a command is useful. So now our aim is to run in terminal each command we have learnt and understand what it does. If you work on them on your own, you can understand file system closely.
Now we will first take a look at how to create new files and directories, so we can create different files and directories, so that we can use our prior knowledge.
I will create some directories in my home directory, you can also create these directories in your home directory for your convenience -

```
totan@Home-Computer ~ $ ls -l
total 63268
drwxr-xr-x  2 totan totan     4096 Jul 15 23:29 bc
drwxr-xr-x  2 totan totan     4096 Oct 18 23:18 Calibre Library
drwxr-xr-x  2 totan totan     4096 Jun 17 19:47 circuits
drwxr-xr-x  3 totan totan     4096 May 20 13:08 data
drwxr-xr-x 10 totan totan     4096 Oct 29 22:03 Desktop
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Documents
drwxr-xr-x  2 totan totan     4096 Jul 30 23:01 Downloads
-rw-r--r--  1 totan totan       15 Oct 27 20:45 file1.txr
-rw-r--r--  1 totan totan 64673787 Aug 24 09:46 file__system
-rw-r--r--  1 totan totan     6955 Jul 30 21:59 Google_Drive_Links.docx
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Music
drwxr-xr-x  2 totan totan     4096 Nov  3 19:30 notes
drwxr-xr-x  2 totan totan     4096 Sep  6 09:37 Pictures
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Public
-rw-r--r--  1 totan totan     1467 Aug  4 20:39 Rahman.m3u
-rw-r--r--  1 totan totan    19350 Oct  8 18:11 shop.pdf
-rw-r--r--  1 totan totan     5614 Oct  8 18:10 shop.sh3d
drwxr-xr-x  2 totan totan     4096 May 20 13:32 source
drwxr-xr-x  2 totan totan     4096 May 20 20:29 Videos
drwxr-xr-x  6 totan totan     4096 Aug 11 08:44 VirtualBox VMs
-rwxrwxrwx  1 totan totan      133 Jun  1 00:40 vnc
totan@Home-Computer ~ $ mkdir dir_creation
totan@Home-Computer ~ $ ls -l
total 63272
drwxr-xr-x  2 totan totan     4096 Jul 15 23:29 bc
drwxr-xr-x  2 totan totan     4096 Oct 18 23:18 Calibre Library
drwxr-xr-x  2 totan totan     4096 Jun 17 19:47 circuits
drwxr-xr-x  3 totan totan     4096 May 20 13:08 data
drwxr-xr-x 10 totan totan     4096 Oct 29 22:03 Desktop
drwxr-xr-x  2 totan totan     4096 Nov  3 19:31 dir_creation
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Documents
drwxr-xr-x  2 totan totan     4096 Jul 30 23:01 Downloads
-rw-r--r--  1 totan totan       15 Oct 27 20:45 file1.txr
-rw-r--r--  1 totan totan 64673787 Aug 24 09:46 file__system
-rw-r--r--  1 totan totan     6955 Jul 30 21:59 Google_Drive_Links.docx
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Music
drwxr-xr-x  2 totan totan     4096 Nov  3 19:30 notes
drwxr-xr-x  2 totan totan     4096 Sep  6 09:37 Pictures
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Public
-rw-r--r--  1 totan totan     1467 Aug  4 20:39 Rahman.m3u
-rw-r--r--  1 totan totan    19350 Oct  8 18:11 shop.pdf
-rw-r--r--  1 totan totan     5614 Oct  8 18:10 shop.sh3d
drwxr-xr-x  2 totan totan     4096 May 20 13:32 source
drwxr-xr-x  2 totan totan     4096 May 20 20:29 Videos
drwxr-xr-x  6 totan totan     4096 Aug 11 08:44 VirtualBox VMs
-rwxrwxrwx  1 totan totan      133 Jun  1 00:40 vnc
totan@Home-Computer ~ $

```

Here I have run a command **mkdir, needless to say that, it is shortcut for '*make directory'. We can create a directory by running this command. We have to type this command followed by a name of our choice, like here I have created a directory **dir_creation. Before and after running the **mkdir command I used the **ls -l command and we have verified that the directory was actually created. You can also create any directory in your home directory with this command, but you need to know some rules for directory or file name -

You can type any character from the keyboard for the file or directory name, except for the forward slash ( **/ ). We already know about [Absolute Paths and Relative Paths](http://unix.palash90.in/2017/07/file-system-absolute-and-relative-path.html). Forward slash is used to isolate the names of two directories, so we can not use it to rename files or directories. 
Also, we use the **/ (forward slash) to indicate the root directory on the Linux/UNIX system, so we can not create a new directory whose name is **/

Some special characters (!, #, $, &,>, <, space) are associated with the shell command, so they can not be typed directly. To type these characters, we have to use escape character. Soon, we will learn about escape character with the help of an example.

On any modern Linux/UNIX system, the name of a file or directory can be from 1 character to 255 characters.

In a directory, we can never create two directories or files of the same name. There can be a file with the same name in two different directories, there is no problem, but there can not be two files with same name within the same directory. Means, the absolute path of any two files can never be the same. Try to justify the first rule using this one. We can not create a file / directory named / because it is already present in the file system.

Following these rules we can create different files and directories. Although Linux/UNIX system allows us to use any combination of characters for file name, in day to day usages we only use a combination of alphabets, underscore(_) and numbers mostly . Modern Linux/UNIX systems also has the ability to use characters from any language in the file or directory name. One important thing to keep in mind is that, it is always good to avoid the use of space as much as possible for file or directory names, because sometimes it becomes horrible to work with that file or directory; every time we have to use escape character to use the file or the directory.
Let us now create some new directories in the directory **dir_creation and ourselves check all rules -![image](https://1.bp.blogspot.com/-GVqhiOVvoyQ/WfyEqU9KPDI/AAAAAAAAJC4/wE1RrzHwOKwNLvt4QcFLnrUhFvqPcpzlwCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-11-03%2B20-29-34.png)

![image](https://1.bp.blogspot.com/-4G_bTZeQRj4/WfyEqT7G24I/AAAAAAAAJC0/UJpyVlYGuY0-UFGyT9_9UN0QqGxIA5aNgCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-11-03%2B20-29-59.png)
**![image](https://1.bp.blogspot.com/-KMAi70m28OQ/WfyEqRd3iSI/AAAAAAAAJCw/w7iETP3yO0sRQcdYqX6X3mwnjo-vLTp5wCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-11-03%2B20-30-14.png)
I wanted to create a directory with name **/ on 3 different Linux/Unix systems, but no one supported me and in every case I was told that the **/ directory is already present. 
**Now we will try to create directories that have special characters in their names -

```
totan@Home-Computer ~/dir_creation $ mkdir ~
mkdir: cannot create directory ‘/home/totan’: File exists
totan@Home-Computer ~/dir_creation $ mkdir ~kj
totan@Home-Computer ~/dir_creation $ ls -l
total 4
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
totan@Home-Computer ~/dir_creation $ mkdir !
totan@Home-Computer ~/dir_creation $ ls -l
total 8
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
totan@Home-Computer ~/dir_creation $ mkdir !d
mkdir daemon-sync
totan@Home-Computer ~/dir_creation $ ls -l
total 12
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 daemon-sync
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
totan@Home-Computer ~/dir_creation $ mkdir !kj
bash: !kj: event not found
totan@Home-Computer ~/dir_creation $ mkdir @l
totan@Home-Computer ~/dir_creation $ ls -l
total 16
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 daemon-sync
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 @l
totan@Home-Computer ~/dir_creation $ mkdir #
mkdir: missing operand
Try 'mkdir --help' for more information.
totan@Home-Computer ~/dir_creation $ mkdir #hg
mkdir: missing operand
Try 'mkdir --help' for more information.
totan@Home-Computer ~/dir_creation $ mkdir $kj
mkdir: missing operand
Try 'mkdir --help' for more information.
totan@Home-Computer ~/dir_creation $ mkdir %
totan@Home-Computer ~/dir_creation $ ls -l
total 20
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 daemon-sync
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 @l
totan@Home-Computer ~/dir_creation $ mkdir %lk
totan@Home-Computer ~/dir_creation $ ls -l
total 24
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 daemon-sync
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 @l
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %lk
totan@Home-Computer ~/dir_creation $ mkdir ^
totan@Home-Computer ~/dir_creation $ ls -l
total 28
drwxr-xr-x 2 totan totan 4096 Nov  3 21:13 ^
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 daemon-sync
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 @l
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %lk
totan@Home-Computer ~/dir_creation $ mkdir &
[1] 18680
totan@Home-Computer ~/dir_creation $ mkdir: missing operand
Try 'mkdir --help' for more information.

[1]+  Exit 1                  mkdir
totan@Home-Computer ~/dir_creation $ mkdir &j
[1] 18681
mkdir: missing operand
Try 'mkdir --help' for more information.
j: command not found
[1]+  Exit 1                  mkdir
totan@Home-Computer ~/dir_creation $ mkdir *
mkdir: cannot create directory ‘^’: File exists
mkdir: cannot create directory ‘!’: File exists
mkdir: cannot create directory ‘%’: File exists
mkdir: cannot create directory ‘daemon-sync’: File exists
mkdir: cannot create directory ‘~kj’: File exists
mkdir: cannot create directory ‘@l’: File exists
mkdir: cannot create directory ‘%lk’: File exists
totan@Home-Computer ~/dir_creation $ mkdir (
bash: syntax error near unexpected token `newline'
totan@Home-Computer ~/dir_creation $ mkdir )
bash: syntax error near unexpected token `)'
totan@Home-Computer ~/dir_creation $ ls -l
total 28
drwxr-xr-x 2 totan totan 4096 Nov  3 21:13 ^
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 !
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 daemon-sync
drwxr-xr-x 2 totan totan 4096 Nov  3 21:11 ~kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 @l
drwxr-xr-x 2 totan totan 4096 Nov  3 21:12 %lk
totan@Home-Computer ~/dir_creation $

```

Having seen the above output, you have come to the conclusion that these special characters are doing a special job, so using them in the file or directory name is not possible. You may have also guessed that **mkdir ~ does not work, because in Linux/UNIX **~ refers to the home directory. So now the question is, if all the characters can be used in Linux for file names, then how do we create a file or directory with a name in which the special characters are used such as **~,!d,!kj, #, #kj   etc. Although in real world, these type of name is extremely rare or almost nowhere used, for the sake of completeness we will take a look at how to create a file or directory with a name composed of these special characters-

```
totan@Home-Computer ~/dir_creation $ mkdir \~
totan@Home-Computer ~/dir_creation $ ls -l
total 4
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
totan@Home-Computer ~/dir_creation $ mkdir \!d
totan@Home-Computer ~/dir_creation $ ls -l
total 8
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
totan@Home-Computer ~/dir_creation $ mkdir \#lk
totan@Home-Computer ~/dir_creation $ ls -l
total 12
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
totan@Home-Computer ~/dir_creation $ mkdir \$kj
totan@Home-Computer ~/dir_creation $ ls -l
total 16
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
totan@Home-Computer ~/dir_creation $ mkdir \&palash
totan@Home-Computer ~/dir_creation $ ls -l
total 20
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
totan@Home-Computer ~/dir_creation $ mkdir \*
totan@Home-Computer ~/dir_creation $ ls -l
total 24
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
totan@Home-Computer ~/dir_creation $ mkdir \(
totan@Home-Computer ~/dir_creation $ ls -l
total 28
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 (
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
totan@Home-Computer ~/dir_creation $ mkdir \)
totan@Home-Computer ~/dir_creation $ ls -l
total 32
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 (
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 )
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
totan@Home-Computer ~/dir_creation $ mkdir \<
totan@Home-Computer ~/dir_creation $ ls -l
total 36
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 <
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 (
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 )
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
totan@Home-Computer ~/dir_creation $ mkdir \>
totan@Home-Computer ~/dir_creation $ ls -l
total 40
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 <
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 >
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 (
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 )
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
totan@Home-Computer ~/dir_creation $ mkdir Palash\ Creating\ Directory\ with\ name\ having\ space
totan@Home-Computer ~/dir_creation $ ls -l
total 44
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 <
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 >
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 (
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 )
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 Palash Creating Directory with name having space
totan@Home-Computer ~/dir_creation $
totan@Home-Computer ~/dir_creation $ mkdir পরশ
totan@Home-Computer ~/dir_creation $ ls -l
total 52
drwxr-xr-x 2 totan totan 4096 Nov  3 21:25 ~
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 <
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 >
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 (
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 )
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 *
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 !d
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 $kj
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 #lk
drwxr-xr-x 2 totan totan 4096 Nov  3 21:26 &palash
drwxr-xr-x 2 totan totan 4096 Nov  3 21:27 Palash Creating Directory with name having space
drwxr-xr-x 2 totan totan 4096 Nov  3 21:29 পরশ

```

Observe carefully, whenever we want to use a special character in the name, we use back slash ( **\ ) before that special character. This backward slash ( **\ ) is called escape character. Using this, we can use special characters in file or directory name. Also, I have created a directory whose name consists of characters from Bengali alphabet.

Now, I will try to create a file whose file name contains 256 characters, you can try to create a file with more number of characters as well -

```
totan@Home-Computer ~/dir_creation $ ls -l
total 0
totan@Home-Computer ~/dir_creation $ touch l6ecYSSPVBgKxMPKQ0k4D1NW8KX8w0JhMTfHYc018gKMfCHsU7rVNQvkazV1Iebvn6PthoSTsU4V3eqvwYLvNhQtbk0PneNavb2HxnogVn8jITHHhReIJWYh257czG0xZ0rS4yUGfDZ1LT8TWli6KO9KT4q48vDZuEPVhYwez7pDTO87EW6snv76LfEHMTALJR7sOv0hymp0ecz2HVgoKWOI98bckXOjWRUet3hjRVWJc0mqfoSFIr6FuXl25Ln2
touch: cannot touch 'l6ecYSSPVBgKxMPKQ0k4D1NW8KX8w0JhMTfHYc018gKMfCHsU7rVNQvkazV1Iebvn6PthoSTsU4V3eqvwYLvNhQtbk0PneNavb2HxnogVn8jITHHhReIJWYh257czG0xZ0rS4yUGfDZ1LT8TWli6KO9KT4q48vDZuEPVhYwez7pDTO87EW6snv76LfEHMTALJR7sOv0hymp0ecz2HVgoKWOI98bckXOjWRUet3hjRVWJc0mqfoSFIr6FuXl25Ln2': File name too long
totan@Home-Computer ~/dir_creation $
totan@Home-Computer ~/dir_creation $
totan@Home-Computer ~/dir_creation $
totan@Home-Computer ~/dir_creation $
totan@Home-Computer ~/dir_creation $ ls -l
total 0
totan@Home-Computer ~/dir_creation $ touch l6ecYSSPVBgKxMPKQ0k4D1NW8KX8w0JhMTfHYc018gKMfCHsU7rVNQvkazV1Iebvn6PthoSTsU4V3eqvwYLvNhQtbk0PneNavb2HxnogVn8jITHHhReIJWYh257czG0xZ0rS4yUGfDZ1LT8TWli6KO9KT4q48vDZuEPVhYwez7pDTO87EW6snv76LfEHMTALJR7sOv0hymp0ecz2HVgoKWOI98bckXOjWRUet3hjRVWJc0mqfoSFIr6FuXl25Ln
totan@Home-Computer ~/dir_creation $
totan@Home-Computer ~/dir_creation $ ls -l
total 0
-rw-r--r-- 1 totan totan 0 Nov  3 20:57 l6ecYSSPVBgKxMPKQ0k4D1NW8KX8w0JhMTfHYc018gKMfCHsU7rVNQvkazV1Iebvn6PthoSTsU4V3eqvwYLvNhQtbk0PneNavb2HxnogVn8jITHHhReIJWYh257czG0xZ0rS4yUGfDZ1LT8TWli6KO9KT4q48vDZuEPVhYwez7pDTO87EW6snv76LfEHMTALJR7sOv0hymp0ecz2HVgoKWOI98bckXOjWRUet3hjRVWJc0mqfoSFIr6FuXl25Ln
totan@Home-Computer ~/dir_creation $

```

See, the first time the number of characters was 256 for directory name, so Linux showed us  touch: can not touch 'l6ecYSSPVBgKxMPKQ0k4D1 ...  but whenever I removed the last letter, Linux has created the file in the directory. In this case you need to know, **touch is a command that will create an empty file for you. After typing the **touch, typing any file name will create an empty file, and if the file is already present then the time stamp of the file will be changed (soon we will know what is time stamp). 
Now we will see the implementation of the last rule -

```
totan@Home-Computer ~/dir_creation $ ls -l
total 0
totan@Home-Computer ~/dir_creation $ touch a
totan@Home-Computer ~/dir_creation $ ls -l
total 0
-rw-r--r-- 1 totan totan 0 Nov  3 21:38 a
totan@Home-Computer ~/dir_creation $ mkdir a
mkdir: cannot create directory ‘a’: File exists
totan@Home-Computer ~/dir_creation $ touch 4
totan@Home-Computer ~/dir_creation $ ls -l
total 0
-rw-r--r-- 1 totan totan 0 Nov  3 21:38 4
-rw-r--r-- 1 totan totan 0 Nov  3 21:38 a
totan@Home-Computer ~/dir_creation $ mkdir 4
mkdir: cannot create directory ‘4’: File exists
totan@Home-Computer ~/dir_creation $ touch 6
totan@Home-Computer ~/dir_creation $ ls -l
total 0
-rw-r--r-- 1 totan totan 0 Nov  3 21:38 4
-rw-r--r-- 1 totan totan 0 Nov  3 21:38 6
-rw-r--r-- 1 totan totan 0 Nov  3 21:38 a
totan@Home-Computer ~/dir_creation $ mkdir 6
mkdir: cannot create directory ‘6’: File exists
totan@Home-Computer ~/dir_creation $ mkdir 8
totan@Home-Computer ~/dir_creation $ ls -l
total 4
-rw-r--r-- 1 totan totan    0 Nov  3 21:38 4
-rw-r--r-- 1 totan totan    0 Nov  3 21:38 6
drwxr-xr-x 2 totan totan 4096 Nov  3 21:38 8
-rw-r--r-- 1 totan totan    0 Nov  3 21:38 a
totan@Home-Computer ~/dir_creation $ touch 8
totan@Home-Computer ~/dir_creation $ ls -l
total 4
-rw-r--r-- 1 totan totan    0 Nov  3 21:38 4
-rw-r--r-- 1 totan totan    0 Nov  3 21:38 6
drwxr-xr-x 2 totan totan 4096 Nov  3 21:39 8
-rw-r--r-- 1 totan totan    0 Nov  3 21:38 a
totan@Home-Computer ~/dir_creation $

```

At first I created a file, whose name is **a  and I tried to create a directory of the same name but I was not successful, the system showed me error -  mkdir: can not create directory 'a': File exists . Isn't it strange that we tried to create a directory but it is showing me an error related to file. No, it is not strange, it is just another thing to keep in mind. Everything in a UNIX/Linux system is considered as a file. Directory is a special type of file; a special file that combines some other files or directory.
Coming back to our original discussion, the same thing happened when I tried to create a directory by name 4  and  6 . But when we first created a directory named 8 and then used **touch command to create a file named 8, the system showed us no error. 

You are in a deep thought on why did this happen, seems like the last rule is violated. How come the system is creating a file named 8. You are mistaken, no file has been created with name  8  in the directory. I won't help you to find the answer, notice the output of all the **ls command, then you can understand exactly what happened. 

To find the answer to this question, you would need help of all the topics, those have been discussed here till date. So, it's like a homework session for you. Go, find the answer before I start our next session.

<div class='
