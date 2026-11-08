---
title: "File system facts"
date: "2017-07-16T12:19:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021052944)"
---

**We have seen how we can check files of a directory from another directory, we have seen how we can navigate to different directories in the file systems, we know how we can see files in a directory, the concept of our absolute path and relative path is now not confusing to us anymore. Now we will try to dive deeper. We can save various files in Linux. Although we can save any type of file, Linux does not know what type a file is or what information is saved in a file. Understanding the content of a file is responsibility of the program and represent to the user, which is being used to open the file. For example, we can save pictures, videos, programming files and other type of files in a directory, Linux/UNIX cares least, all it knows is how to store the information safe and how to retrieve from disk.

Although Linux does not even know what is contained in a file, we can have an idea about any file with a program. Name of this special program **file. We will see some examples of this program -

```
totan@Home-Computer /mnt/sda3/Media/E-books/Books $ pwd
/mnt/sda3/Media/E-books/Books
totan@Home-Computer /mnt/sda3/Media/E-books/Books $ ls -al
total 21468
dr-xr-xr-x  3 palash palash     4096 Aug 15  2015 .
dr-xr-xr-x 34 palash palash     4096 Jul 31 08:15 ..
-r-xr-xr-x  1 palash palash 11752144 Aug  9  2008 100.Great.Problems.Of.Elementary.pdf
-r-xr-xr-x  1 palash palash  5227704 Feb 23  2008 learnmatlab.pdf
-r-xr-xr-x  1 palash palash  4871781 Feb 23  2008 MATLAB® Demystifi ed.pdf
-r-xr-xr-x  1 palash palash   109568 Feb 23  2008 Matlab_Tutorial.pdf
dr-xr-xr-x  4 palash palash     4096 Aug 21  2015 Saradindu
totan@Home-Computer /mnt/sda3/Media/E-books/Books $ file 100.Great.Problems.Of.Elementary.pdf
100.Great.Problems.Of.Elementary.pdf: PDF document, version 1.4
totan@Home-Computer /mnt/sda3/Media/E-books/Books $

```

We can navigate to a directory and use this command to see what a file contains and we can use a proper command to open a file. This helps us at times when dealing with different file types or a file whose extension is unknown to us -

```
totan@Home-Computer ~ $ file /mnt/sda3/Media/E-books/Matlab\ Techniques.pdf
/mnt/sda3/Media/E-books/Matlab Techniques.pdf: PDF document, version 1.3
totan@Home-Computer ~ $ file notes/ch1.3.txt
notes/ch1.3.txt: ASCII text, with very long lines
totan@Home-Computer ~ $

```

In the previous example, we used absolute path and relative path to run the **file command; time for some more examples which will strengthen the understanding of absolute path, relative path and **file command as well-

```
totan@Home-Computer ~ $ file ../../../mnt/sda3/Media/E-books/epub/Satyajit\ Ray/Koilash\ Choudhrir\ Pathar\(Feluda\ -\ Satyajit-Roy.epub
../../../mnt/sda3/Media/E-books/epub/Satyajit Ray/Koilash Choudhrir Pathar(Feluda - Satyajit-Roy.epub: EPUB document
totan@Home-Computer ~ $
totan@Home-Computer ~ $ file /mnt/sda3/Media/Audio/Rabindra\ Sangeet/Various\ Artist/AMI\ CHINI\ GO\ CHINI.MP3.mp3
/mnt/sda3/Media/Audio/Rabindra Sangeet/Various Artist/AMI CHINI GO CHINI.MP3.mp3: Audio file with ID3 version 2.4.0, contains: MPEG ADTS, layer III, v1, 128 kbps, 44.1 kHz, JntStereo
totan@Home-Computer ~ $
totan@Home-Computer ~ $ file /mnt/sda3/Media/Vision/mahabharat/39\ Coronation\ of\ Yudhirsthir\,\ Khandavprastha\ becomes\ Indraprastha.avi
/mnt/sda3/Media/Vision/mahabharat/39 Coronation of Yudhirsthir, Khandavprastha becomes Indraprastha.avi: RIFF (little-endian) data, AVI, 640 x 352, 29.97 fps, video: DivX 5, audio: MPEG-1 Layer 3 (stereo, 48000 Hz)
totan@Home-Computer ~ $
totan@Home-Computer ~ $ file /mnt/sda3/Media/Memory\ in\ Cam/Canon\ PowerShot\ A3100\ IS/
/mnt/sda3/Media/Memory in Cam/Canon PowerShot A3100 IS/: directory
totan@Home-Computer ~ $ file /mnt/sda3/Media/Memory\ in\ Cam/Canon\ PowerShot\ A3100\ IS/2011_02_14/IMG_0057.JPG
/mnt/sda3/Media/Memory in Cam/Canon PowerShot A3100 IS/2011_02_14/IMG_0057.JPG: JPEG image data, Exif standard: [TIFF image data, little-endian, direntries=10, description=                               , manufacturer=Canon, model=Canon PowerShot A3100 IS, orientation=upper-left, xresolution=204, yresolution=212, resolutionunit=2, datetime=2011:02:14 00:19:43], baseline, precision 8, 4000x3000, frames 3
totan@Home-Computer ~ $

```

The **file command sometimes not only shows the type of the file content but some details as well, which sometimes proves to be very useful.

Now time to check some more facts about Linux/UNIX File System -

#### File or directory name is case sensitive

That means file1.txt, file1.txt, fIle1.txt, fiLe1.txt, filE1.txt, FILE1.txt are all different files on UNIX/Linux. So if we save a file as file1.txt and want to open as File1.txt, we can not do that. Let's look at an example -

```
totan@Home-Computer ~ $ ls -l
total 63268
drwxr-xr-x  2 totan totan     4096 Jul 15 23:29 bc
drwxr-xr-x  2 totan totan     4096 Oct 18 23:18 Calibre Library
drwxr-xr-x  2 totan totan     4096 Jun 17 19:47 circuits
drwxr-xr-x  3 totan totan     4096 May 20 13:08 data
drwxr-xr-x 10 totan totan     4096 Oct 26 23:10 Desktop
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Documents
drwxr-xr-x  2 totan totan     4096 Jul 30 23:01 Downloads
-rw-r--r--  1 totan totan       15 Oct 27 20:45 file1.txr
-rw-r--r--  1 totan totan 64673787 Aug 24 09:46 file__system
-rw-r--r--  1 totan totan     6955 Jul 30 21:59 Google_Drive_Links.docx
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Music
drwxr-xr-x  2 totan totan     4096 Sep  6 21:44 notes
drwxr-xr-x  2 totan totan     4096 Sep  6 09:37 Pictures
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Public
-rw-r--r--  1 totan totan     1467 Aug  4 20:39 Rahman.m3u
-rw-r--r--  1 totan totan    19350 Oct  8 18:11 shop.pdf
-rw-r--r--  1 totan totan     5614 Oct  8 18:10 shop.sh3d
drwxr-xr-x  2 totan totan     4096 May 20 13:32 source
drwxr-xr-x  2 totan totan     4096 May 20 20:29 Videos
drwxr-xr-x  6 totan totan     4096 Aug 11 08:44 VirtualBox VMs
-rwxrwxrwx  1 totan totan      133 Jun  1 00:40 vnc
totan@Home-Computer ~ $ less File1.txr
File1.txr: No such file or directory
totan@Home-Computer ~ $

```

Notice that  there is a file named file1.txr in my home directory  but when I tried to open the file by typing ***less ***File1.txr , Linux showed me an error and explained to us that the file I want to open is not in that directory.

#### The Linux system does not necessitate a file to have an extension

The extension is the part of the file name from the last **. to the rest of the last of the file name. Let's check some files and there extensions -

```

dr-xr-xr-x  2 palash palash     4096 Aug 15  2015 Stellar Astrophysics
dr-xr-xr-x  2 palash palash     4096 Aug 15  2015 The Hackers Underground Handbook
-r-xr-xr-x  1 palash palash   202453 May  2  2011 The Pythagorean Theorem.pdf
-r-xr-xr-x  1 palash palash 10477660 Mar 15  2011 ebooksclub.org__Modern_Graph_Theory.djvu

```

Here Stellar Astrophysics,   The Hackers Underground Handbook are directories, the rest are files and the  extension of the Pythagorean Theorem.pdf  file is  .pdf  and  ebooksclub.org__Modern_Graph_Theory.djvu  file extension is .djvu

In many cases, extensions can be used to understand the file and run the required commands on the file, but the file system does not mandate a file to have extension, there may not be any extension of the file. For example, there is no extension of the files below -

```
-rw-r--r--  1 totan totan        2 Oct 27 20:59 ch1
-rw-r--r--  1 totan totan        4 Oct 27 21:00 ch2
-rw-r--r--  1 totan totan        4 Oct 27 21:00 ch3
-r-xr-xr-x  1 totan totan 29436482 Oct 27 21:00 ch4

```

But the amazing thing is that despite the file extension is missing, we can run the **file command and know what kind of file it can be-

```
totan@Home-Computer ~ $ file ch1
ch1: ASCII text
totan@Home-Computer ~ $ file ch2
ch2: ASCII text
totan@Home-Computer ~ $ file ch3
ch3: ASCII text
totan@Home-Computer ~ $ file ch4
ch4: EPUB document
totan@Home-Computer ~ $

```

#### Hidden file

#### If the first character of a file or directory name is .  in UNIX/Linux system it is called a hidden file, and generally these files are not visible. We have to use another method to check the hidden files (we will see how they are done). Usually hidden files serve the purpose of security and prevents unnecessary damage to system or configuration files. Generally any Linux system has many configurations, on which the system runs, so if a normal user changes or deletes these files, the system will not work properly, so these files are typically hidden files. Although there are other methods for file security, hidden files are also useful.
Besides security, there is another reason behind it, all the files are not always necessary, it is difficult to find the required files if every time system shows us all files. Let's see what are the files present in my home directory -

```
drwxr-xr-x  2 totan totan     4096 Jul 15 23:29 bc
drwxr-xr-x  2 totan totan     4096 Oct 18 23:18 Calibre Library
drwxr-xr-x  2 totan totan     4096 Jun 17 19:47 circuits
drwxr-xr-x  3 totan totan     4096 May 20 13:08 data
drwxr-xr-x 10 totan totan     4096 Oct 26 23:10 Desktop
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Documents
drwxr-xr-x  2 totan totan     4096 Jul 30 23:01 Downloads
-rw-r--r--  1 totan totan       15 Oct 27 20:45 file1.txr
-rw-r--r--  1 totan totan 64673787 Aug 24 09:46 file__system
-rw-r--r--  1 totan totan     6955 Jul 30 21:59 Google_Drive_Links.docx
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Music
drwxr-xr-x  2 totan totan     4096 Sep  6 21:44 notes
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

This is the normal view of my home directory, now we'll see all files and directories in my home directory including hidden files-

```
drwxr-xr-x  4 root  root      4096 May 15 00:04 ..
-rw-------  1 totan totan    12288 Sep  7 09:01 .abc.swp
drwx------  3 totan totan     4096 May 23 19:12 .adobe
drwxr-xr-x  4 totan totan     4096 Jul 30 22:50 .audacity-data
-rw-------  1 totan totan    26230 Oct 26 23:10 .bash_history
-rw-r--r--  1 totan totan      220 May 15 00:04 .bash_logout
-rw-r--r--  1 totan totan     4031 Jul 31 09:10 .bashrc
drwxr-xr-x  2 totan totan     4096 Jul 15 23:29 bc
drwx------ 26 totan totan     4096 Sep 21 15:46 .cache
-rw-------  1 totan totan       27 Jul 17 08:34 .calc_history
drwxr-xr-x  2 totan totan     4096 Oct 18 23:18 Calibre Library
-rw-r--r--  1 totan totan      178 Jun  1 23:49 .candidate-selections.json
drwxr-xr-x  4 totan totan     4096 Oct 27 19:40 .cinnamon
drwxr-xr-x  2 totan totan     4096 Jun 17 19:47 circuits
drwxr-xr-x 35 totan totan     4096 Oct 18 17:35 .config
drwxr-xr-x  3 totan totan     4096 May 20 13:08 data
drwx------  3 totan totan     4096 May 15 00:14 .dbus
drwxr-xr-x 10 totan totan     4096 Oct 26 23:10 Desktop
-rw-------  1 totan totan       29 Oct 27 19:39 .dmrc
-rw-------  1 totan totan       29 May 25 21:55 .dmrc.03NT0Y
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Documents
drwx------  2 totan totan     4096 Jun 15 18:35 .dosbox
drwxr-xr-x  2 totan totan     4096 Jul 30 23:01 Downloads
drwxr-xr-x  2 totan totan     4096 Jun  2 16:25 .dreamchess
drwxr-xr-x  3 totan totan     4096 May 23 22:11 .dvdcss
-rw-------  1 totan totan       16 Jun  2 16:23 .esd_auth
drwxr-xr-x  3 totan totan     4096 Oct  8 17:26 .eteks
drwxr-xr-x  2 totan totan     4096 Oct 18 23:07 .FBReader
-rw-r--r--  1 totan totan       15 Oct 27 20:45 file1.txr
-rw-r--r--  1 totan totan 64673787 Aug 24 09:46 file__system
drwx------  3 totan totan     4096 Oct 24 21:35 .gconf
drwxr-xr-x 24 totan totan     4096 Aug  4 22:58 .gimp-2.8
drwx------  3 totan totan     4096 May 15 00:23 .gnome
drwx------  3 totan totan     4096 May 28 00:01 .gnome2
drwx------  2 totan totan     4096 May 23 15:18 .gnome2_private
-rw-r--r--  1 totan totan     6955 Jul 30 21:59 Google_Drive_Links.docx
-rw-------  1 totan totan    84114 Oct 27 19:39 .ICEauthority
drwxr-xr-x  4 totan totan     4096 Jun 15 20:40 .java
-rw-------  1 totan totan       86 Jul 13 08:56 .lesshst
drwx------  2 totan totan     4096 Aug 20 13:17 .links2
drwxr-xr-x  5 totan totan     4096 May 21 19:51 .linuxmint
drwxr-xr-x  3 totan totan     4096 May 15 00:14 .local
drwx------  3 totan totan     4096 May 23 19:12 .macromedia
drwxr-xr-x  4 totan totan     4096 May 21 19:56 .mozilla
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Music
drwxr-xr-x  2 totan totan     4096 Jul  6 22:08 .nano
drwxr-xr-x  2 totan totan     4096 Sep  6 21:44 notes
drwxr-xr-x  2 totan totan     4096 May 24 15:54 .oracle_jre_usage
drwxr-xr-x  2 totan totan     4096 Sep  6 09:37 Pictures
drwx------  3 totan totan     4096 May 15 00:23 .pki
-rw-r--r--  1 totan totan      655 May 15 00:04 .profile
drwxr-xr-x  2 totan totan     4096 May 15 00:14 Public
-rw-------  1 totan totan       11 Jul 12 09:49 .python_history
-rw-r--r--  1 totan totan     1467 Aug  4 20:39 Rahman.m3u
-rw-------  1 totan totan     1024 Jun 15 08:15 .rnd
-rw-r--r--  1 totan totan    19350 Oct  8 18:11 shop.pdf
-rw-r--r--  1 totan totan     5614 Oct  8 18:10 shop.sh3d
drwxr-xr-x  2 totan totan     4096 May 20 13:32 source
drwx------  2 totan totan     4096 May 28 00:01 .stardict
-rw-------  1 totan totan    12288 Jun 15 06:55 .swo
-rw-------  1 totan totan    49152 May 23 21:25 .swp
drwx------  4 totan totan     4096 Jul  6 08:49 .thunderbird
drwxr-xr-x  4 totan totan     4096 Jun  4 18:22 .tor-browser-en
drwxr-xr-x  2 totan totan     4096 May 20 20:29 Videos
drwxr-xr-x  2 totan totan     4096 Aug 21 12:56 .vim
-rw-------  1 totan totan    14417 Oct 27 21:00 .viminfo
drwxr-xr-x  6 totan totan     4096 Aug 11 08:44 VirtualBox VMs
-rwxrwxrwx  1 totan totan      133 Jun  1 00:40 vnc
drwxr-xr-x  2 totan totan     4096 Jun  4 00:36 .vnc
-rw-------  1 root  root       445 May 21 19:42 .Xauthority
-rw-------  2 totan totan        0 May 21 19:43 .Xauthority-c
-rw-------  2 totan totan        0 May 21 19:43 .Xauthority-l
drwxr-xr-x  3 totan totan     4096 Jul 28 23:47 .xdm
-rw-r--r--  1 totan totan      131 Jun  1 23:03 .xinputrc
-rw-r--r--  1 totan totan     4285 Oct 27 21:15 .xsession-errors
-rw-r--r--  1 totan totan    39533 Jun 10 22:30 .zcompdump
-rw-------  1 totan totan       27 Jul  8 19:43 .zsh_history
-rw-r--r--  1 totan totan     1295 Jul  8 18:18 .zshrc
totan@Home-Computer ~ $
```

Looking at the difference between the two outputs, you can understand why hidden files are important for day to day use.

Next we will take a look at the other file system commands and check what more we can do with them.

<div class='
