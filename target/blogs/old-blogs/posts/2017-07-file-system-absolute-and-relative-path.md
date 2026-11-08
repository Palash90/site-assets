---
title: "File system - Absolute and Relative path"
date: "2017-07-16T12:20:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021051902)"
---

We have already seen how to navigate the file system and how we can easily navigate to the parent directory. Also we have seen, we do not have to navigate from **root(/) directory to navigate to the parent directory. Now we will see how we can go from one directory to another directory without starting navigation from the **root(/) directory.
<u>
<u>Absolute Path  - We have seen that we can navigate from **root(/) directory to any directory by navigating through its branch. Any directory on the Linux / Unix system is actually child of the root directory. So when we run the **pwd command, the output starts from **root**(/) directory all the time. This output is the directory's absolute path. This means that the path begins with **root(/) and navigates up to that directory. Let's check two different outputs from Linux Mint and Free BSD of the **pwd command of different directories -

**Free BSD

![image](https://2.bp.blogspot.com/-OvYGo8f-4S4/Wa9k5_fy2rI/AAAAAAAAHi8/C-vfYfn6trgYnDAR2S7PTDl6Lg-JrDtQACLcBGAs/s1600/Screenshot%2Bfrom%2B2017-09-06%2B08-30-17.png)
**Linux Mint

```
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer /mnt/sda3/Media/Audio/A R Rahman $ pwd
/mnt/sda3/Media/Audio/A R Rahman

```

Knowing the absolute path of a directory, we can easily navigate to that directory from any directory. With an example we will try to understand this -

```
totan@Home-Computer ~/notes $ pwd
/home/totan/notes
totan@Home-Computer ~/notes $ cd /mnt/sda3/Media/Audio/Bengali/Balam/
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer ~/notes $ cd /home/totan/notes
totan@Home-Computer ~/notes $ pwd
/home/totan/notes
totan@Home-Computer ~/notes $ cd /home/totan/notes
totan@Home-Computer ~/notes $ pwd
/home/totan/notes

```

<u style="font-weight: bold;">Relative Path**  - With absolute Path We can easily navigate from any directory to any other directory, but it is very difficult to remember or type the large directory paths all the time, so we will look at the other way to navigate the file system. **Suppose I have heard the song 'Nupur Baje' of Balam from /mnt/sda3/media/audio/Bengali/ directory and now want to listen to 'Latika's Theme' from Slumdog Millionaire movie, then I need to navigate to the directory as follows-

```
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ cd /mnt/sda3/Media/Audio/A\ R\ Rahman/Slumdog\ Millionaire/
totan@Home-Computer /mnt/sda3/Media/Audio/A R Rahman/Slumdog Millionaire $

```

Typing the absolute path for navigating directories is very difficult and time-consuming and in reality, we need not to navigate to a directory from the **root directory, so we will learn about relative path for our convenience. The relative path means the path that navigates from the current directory and navigates to another directory. Let's look at the previous example but this time with relative path-

```
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ cd ../../A\ R\ Rahman/Slumdog\ Millionaire/
totan@Home-Computer /mnt/sda3/Media/Audio/A R Rahman/Slumdog Millionaire $ pwd
/mnt/sda3/Media/Audio/A R Rahman/Slumdog Millionaire
totan@Home-Computer /mnt/sda3/Media/Audio/A R Rahman/Slumdog Millionaire $

```

First we were at  /mnt/sda3/Media/Audio/Bengali/Balam  directory and from there we would like to go to  /mnt/sda3/Media/Audio/A R Rahman/Slumdog Millionaire directory. Now notice, the parent of the Balam directory is Bengali, whose parent is Audio, again the Audio directory's one child is A R Rahman, whose child is Slumdog Millionaire. In other words Balam directory and Slumdog Millionaire directory's ancestor is Audio directory, we used this relation and we have already learned that if we type **cd .. we can navigate to the parent directory. Using these information, we formed the relative path. Now, we'll describe how the relative path works. So the first .. of our command will take us to Bengali directory, from there the **.. will take us to the Audio directory, from there we will navigate to the directory A\ R\ Rahman. If you don't understand A\  R\  Rahman, then know that A\  R\  Rahman means A R Rahman in our case, we will discuss this later in detail. We now navigate from the A R Rahman directory to the Slumdog Millionaire directory, where my favourite soundtrack is present. 

Now we will take a look at another type of relative path related to navigation, where we do not navigate to a parent directory, rather we are navigating to the child of current directory -

```
totan@Home-Computer /mnt/sda3/Media $ pwd
/mnt/sda3/Media
totan@Home-Computer /mnt/sda3/Media $ cd ./Audio/Bengali/Balam/
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $

```

See I've used another shortcut here '**.'. It actually means the current directory. That is, if we have to navigate to the child of the current directory, then we can use ' **. '. 

Without any shortcut also we can navigate child directories -

```
totan@Home-Computer /mnt/sda3/Media $ pwd
/mnt/sda3/Media
totan@Home-Computer /mnt/sda3/Media $ cd Audio/Bengali/Balam/
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $

```

That is, if we want to navigate to the child directory of any directory, we will navigate to that directory by just typing the command **cd <directory_name>. The question might arise in your mind, if we can navigate child directories without any shortcut, then what is it used for **./. To get an answer to this question, you have to wait a little longer. For now, know that both are used and you can use any one as per your requirement or convenience.

Let us now see some examples related to navigation (using both Absolute Path and Relative Path) - 
**Free BSD
![image](https://1.bp.blogspot.com/-k1Obg7Bdpjo/Wa90nwmGzxI/AAAAAAAAHjg/cWfGVdPOoH0MAU1iIqj255NgqqdiFpOgwCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-09-06%2B09-37-34.png)

**Linux Mint

```
totan@Home-Computer ~ $ cd ~
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ cd /
totan@Home-Computer / $ pwd
/
totan@Home-Computer / $ cd /boot
totan@Home-Computer /boot $ pwd
/boot
totan@Home-Computer /boot $ cd ./grub
totan@Home-Computer /boot/grub $ pwd
/boot/grub
totan@Home-Computer /boot/grub $ cd ../../bin
totan@Home-Computer /bin $ pwd
/bin
totan@Home-Computer /bin $ cd ../home/totan/notes/
totan@Home-Computer ~/notes $ pwd
/home/totan/notes
totan@Home-Computer ~/notes $ cd ../Pictures/
totan@Home-Computer ~/Pictures $ pwd
/home/totan/Pictures
totan@Home-Computer ~/Pictures $ cd /mnt/sda3/Media/E-books/
totan@Home-Computer /mnt/sda3/Media/E-books $ pwd
/mnt/sda3/Media/E-books
totan@Home-Computer /mnt/sda3/Media/E-books $ cd ../Audio/Bengali/Balam/
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ cd ..
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali $ cd ./Balam/
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ cd /mnt/sda3/Media/Audio/Bengali/Balam/
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $ pwd
/mnt/sda3/Media/Audio/Bengali/Balam
totan@Home-Computer /mnt/sda3/Media/Audio/Bengali/Balam $

```

Actually, we can navigate to the same directory on the Linux / Unix system in many ways, hopefully this is a lot clearer in the last example. Not just navigating, with the help of Absolute and Relative Path, we can run any command on any directory from any directory. We will check that with the **ls -l command by running it on any other directory from our home directory -

```
totan@Home-Computer ~ $ cd ~
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ ls -l /
total 136
drwxr-xr-x   2 root root 12288 Jul 30 21:08 bin
drwxr-xr-x   3 root root  4096 Sep  1 08:48 boot
drwxr-xr-x   2 root root  4096 Jan 10  2017 cdrom
drwxr-xr-x  22 root root  4340 Sep  6 08:10 dev
drwxr-xr-x 162 root root 12288 Sep  6 08:13 etc
drwxr-xr-x   4 root root  4096 May 15 00:04 home
lrwxrwxrwx   1 root root    32 Sep  1 08:47 initrd.img -> boot/initrd.img-4.4.0-93-generic
lrwxrwxrwx   1 root root    32 Aug 20 12:11 initrd.img.old -> boot/initrd.img-4.4.0-92-generic
drwxr-xr-x  25 root root  4096 Jun 20 20:20 lib
drwxr-xr-x   2 root root 12288 Jun 20 20:20 lib32
drwxr-xr-x   2 root root  4096 Jun 20 20:20 lib64
drwx------   2 root root 16384 Jan 10  2017 lost+found
drwxr-xr-x   4 root root  4096 May 31 22:58 media
drwxr-xr-x   4 root root  4096 Jan 11  2017 mnt
drwxr-xr-x   8 root root  4096 Jun  2 20:42 opt
dr-xr-xr-x 223 root root     0 Sep  6 08:10 proc
drwx------  19 root root  4096 Jul 29 00:01 root
drwxr-xr-x  33 root root  1100 Sep  6 08:46 run
drwxr-xr-x   2 root root 12288 Jul 30 21:08 sbin
drwxr-xr-x   2 root root  4096 Dec 14  2016 srv
dr-xr-xr-x  13 root root     0 Sep  6 08:10 sys
drwxrwxrwt  14 root root 20480 Sep  6 09:40 tmp
drwxr-xr-x   3 root root  4096 Jan 12  2017 u01
drwxr-xr-x  11 root root  4096 Jun 16 20:47 usr
drwxr-xr-x  11 root root  4096 Dec 14  2016 var
lrwxrwxrwx   1 root root    29 Sep  1 08:47 vmlinuz -> boot/vmlinuz-4.4.0-93-generic
lrwxrwxrwx   1 root root    29 Aug 20 12:11 vmlinuz.old -> boot/vmlinuz-4.4.0-92-generic
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ ls -l /boot
total 779672
-rw-r--r-- 1 root root  1243479 Dec  3  2016 abi-4.4.0-53-generic
-rw-r--r-- 1 root root  1244118 Jan  7  2017 abi-4.4.0-59-generic
-rw-r--r-- 1 root root  1244118 Jan 18  2017 abi-4.4.0-62-generic
-rw-r--r-- 1 root root  1245512 Feb  2  2017 abi-4.4.0-63-generic
-rw-r--r-- 1 root root  1245512 Feb 20  2017 abi-4.4.0-64-generic
-rw-r--r-- 1 root root  1245512 Mar  3  2017 abi-4.4.0-66-generic
-rw-r--r-- 1 root root  1246312 Apr 27 23:54 abi-4.4.0-78-generic
-rw-r--r-- 1 root root  1246311 May 18 03:39 abi-4.4.0-79-generic
-rw-r--r-- 1 root root  1246311 Jun 14 17:54 abi-4.4.0-81-generic
-rw-r--r-- 1 root root  1246511 Jun 27 01:15 abi-4.4.0-83-generic
-rw-r--r-- 1 root root  1246670 Jul 18 20:30 abi-4.4.0-87-generic
-rw-r--r-- 1 root root  1246835 Aug  1 03:55 abi-4.4.0-89-generic
-rw-r--r-- 1 root root  1246835 Aug  8 19:28 abi-4.4.0-91-generic
-rw-r--r-- 1 root root  1246835 Aug 10 16:32 abi-4.4.0-92-generic
-rw-r--r-- 1 root root  1247269 Aug 12 05:10 abi-4.4.0-93-generic
-rw-r--r-- 1 root root   189877 Dec  3  2016 config-4.4.0-53-generic
-rw-r--r-- 1 root root   190047 Jan  7  2017 config-4.4.0-59-generic
-rw-r--r-- 1 root root   190047 Jan 18  2017 config-4.4.0-62-generic
-rw-r--r-- 1 root root   190247 Feb  2  2017 config-4.4.0-63-generic
-rw-r--r-- 1 root root   190247 Feb 20  2017 config-4.4.0-64-generic
-rw-r--r-- 1 root root   190247 Mar  3  2017 config-4.4.0-66-generic
-rw-r--r-- 1 root root   190355 Apr 27 23:54 config-4.4.0-78-generic
-rw-r--r-- 1 root root   190356 May 18 03:39 config-4.4.0-79-generic
-rw-r--r-- 1 root root   190356 Jun 14 17:54 config-4.4.0-81-generic
-rw-r--r-- 1 root root   190356 Jun 27 01:15 config-4.4.0-83-generic
-rw-r--r-- 1 root root   190356 Jul 18 20:30 config-4.4.0-87-generic
-rw-r--r-- 1 root root   190356 Aug  1 03:55 config-4.4.0-89-generic
-rw-r--r-- 1 root root   190356 Aug  8 19:28 config-4.4.0-91-generic
-rw-r--r-- 1 root root   190356 Aug 10 16:32 config-4.4.0-92-generic
-rw-r--r-- 1 root root   190356 Aug 12 05:10 config-4.4.0-93-generic
drwxr-xr-x 5 root root     4096 Sep  1 08:48 grub
-rw-r--r-- 1 root root 40764194 Jul  6 09:50 initrd.img-4.4.0-53-generic
-rw-r--r-- 1 root root 40765047 Jul  6 09:50 initrd.img-4.4.0-59-generic
-rw-r--r-- 1 root root 40770453 Jul  6 09:50 initrd.img-4.4.0-62-generic
-rw-r--r-- 1 root root 40751691 Jul  6 09:49 initrd.img-4.4.0-63-generic
-rw-r--r-- 1 root root 40756164 Jul  6 09:49 initrd.img-4.4.0-64-generic
-rw-r--r-- 1 root root 40756852 Jul  6 09:49 initrd.img-4.4.0-66-generic
-rw-r--r-- 1 root root 40763065 Jul  6 09:48 initrd.img-4.4.0-78-generic
-rw-r--r-- 1 root root 40764187 Jul  6 09:48 initrd.img-4.4.0-79-generic
-rw-r--r-- 1 root root 40763389 Jul  6 09:48 initrd.img-4.4.0-81-generic
-rw-r--r-- 1 root root 40763060 Jul 23 10:44 initrd.img-4.4.0-83-generic
-rw-r--r-- 1 root root 40782543 Jul 25 09:09 initrd.img-4.4.0-87-generic
-rw-r--r-- 1 root root 40787197 Aug  4 21:12 initrd.img-4.4.0-89-generic
-rw-r--r-- 1 root root 40783402 Aug 11 20:36 initrd.img-4.4.0-91-generic
-rw-r--r-- 1 root root 40786061 Aug 20 12:11 initrd.img-4.4.0-92-generic
-rw-r--r-- 1 root root 40793891 Sep  1 08:48 initrd.img-4.4.0-93-generic
-rw-r--r-- 1 root root   182704 Jan 28  2016 memtest86+.bin
-rw-r--r-- 1 root root   184380 Jan 28  2016 memtest86+.elf
-rw-r--r-- 1 root root   184840 Jan 28  2016 memtest86+_multiboot.bin
-rw------- 1 root root  3874377 Dec  3  2016 System.map-4.4.0-53-generic
-rw------- 1 root root  3875594 Jan  7  2017 System.map-4.4.0-59-generic
-rw------- 1 root root  3875553 Jan 18  2017 System.map-4.4.0-62-generic
-rw------- 1 root root  3883990 Feb  2  2017 System.map-4.4.0-63-generic
-rw------- 1 root root  3883990 Feb 20  2017 System.map-4.4.0-64-generic
-rw------- 1 root root  3883990 Mar  3  2017 System.map-4.4.0-66-generic
-rw------- 1 root root  3882872 Apr 27 23:54 System.map-4.4.0-78-generic
-rw------- 1 root root  3883279 May 18 03:39 System.map-4.4.0-79-generic
-rw------- 1 root root  3883391 Jun 14 17:54 System.map-4.4.0-81-generic
-rw------- 1 root root  3883887 Jun 27 01:15 System.map-4.4.0-83-generic
-rw------- 1 root root  3884173 Jul 18 20:30 System.map-4.4.0-87-generic
-rw------- 1 root root  3884798 Aug  1 03:55 System.map-4.4.0-89-generic
-rw------- 1 root root  3884798 Aug  8 19:28 System.map-4.4.0-91-generic
-rw------- 1 root root  3884798 Aug 10 16:32 System.map-4.4.0-92-generic
-rw------- 1 root root  3885811 Aug 12 05:10 System.map-4.4.0-93-generic
-rw-r--r-- 1 root root  7065648 Dec 14  2016 vmlinuz-4.4.0-53-generic
-rw------- 1 root root  7069136 Jan  7  2017 vmlinuz-4.4.0-59-generic
-rw------- 1 root root  7070992 Jan 18  2017 vmlinuz-4.4.0-62-generic
-rw------- 1 root root  7087088 Feb  2  2017 vmlinuz-4.4.0-63-generic
-rw------- 1 root root  7087152 Feb 20  2017 vmlinuz-4.4.0-64-generic
-rw------- 1 root root  7087024 Mar  3  2017 vmlinuz-4.4.0-66-generic
-rw------- 1 root root  7089552 Apr 27 23:54 vmlinuz-4.4.0-78-generic
-rw------- 1 root root  7091696 May 18 03:39 vmlinuz-4.4.0-79-generic
-rw------- 1 root root  7092784 Jun 14 17:54 vmlinuz-4.4.0-81-generic
-rw------- 1 root root  7092720 Jun 27 01:15 vmlinuz-4.4.0-83-generic
-rw------- 1 root root  7095888 Jul 18 20:30 vmlinuz-4.4.0-87-generic
-rw------- 1 root root  7098096 Aug  1 03:55 vmlinuz-4.4.0-89-generic
-rw------- 1 root root  7097936 Aug  8 19:28 vmlinuz-4.4.0-91-generic
-rw------- 1 root root  7098032 Aug 10 16:32 vmlinuz-4.4.0-92-generic
-rw------- 1 root root  7097296 Aug 12 05:10 vmlinuz-4.4.0-93-generic
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ ls -l ./notes
total 44
-rw-r--r-- 1 totan totan  3066 Jul 25 09:55 ch1.1.txt
-rw-r--r-- 1 totan totan   576 Aug 10 09:08 ch1.2.txt
-rw-r--r-- 1 totan totan  1568 Jun 11 11:06 ch1.3.txt
-rw-r--r-- 1 totan totan 12512 Jun 18 20:24 ch1.4.txt
-rw-r--r-- 1 totan totan   361 Jun 18 20:27 ch1.5.txt
-rw-r--r-- 1 totan totan  1641 Jun 27 22:09 ch2.1.txt
-rw-r--r-- 1 totan totan    10 Aug  9 22:38 commands.txt
-rw-r--r-- 1 totan totan   763 Aug 13 22:13 next_blog.txt
totan@Home-Computer ~ $ ls -l /mnt/sda3/Media/Audio/Bengali/
total 84
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Balam
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Baul Rocks
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 BCZ I LOVE BANGLADESH
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Bengali
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Bengali Band
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Ekhon Ami
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Ekti Tarar Khoje
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Favourite Baul
dr-xr-xr-x 22 palash palash 4096 Aug 15  2015 Movie
dr-xr-xr-x 22 palash palash 4096 Aug 15  2015 Movies
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Prachir
dr-xr-xr-x  8 palash palash 4096 Aug 15  2015 Rabindra Sangeet
dr-xr-xr-x  2 palash palash 4096 Dec 31  2015 Sa Re Ga Ma Pa
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Satyajit Roy
dr-xr-xr-x  3 palash palash 4096 Aug 15  2015 Serials
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 SRIKANTA
dr-xr-xr-x  2 palash palash 4096 Aug 15  2015 Tahsan
-r-xr-xr-x  1 palash palash 8704 Jun 28  2011 Thumbs.db
dr-xr-xr-x  8 palash palash 4096 Aug 15  2015 Various
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ ls -l ../../mnt/sda3/Media/Audio/Bengali/Balam/
total 313052
-r-xr-xr-x 1 palash palash 5123166 Jan 19  2010 01 - Balam Ft. Julee - Aajhor Bristi (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5282810 Jan 19  2010 01 - Balam - Premer Dhun (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8420538 Jan 20  2010 01 - Balam - Prem Shikari (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4234952 Jan 19  2010 02 - Balam Ft. Julee - Tara Guni (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5282794 Jan 19  2010 02 - Balam - Shongi Hobe Ki (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8327464 Jan 20  2010 02 - Julie - Meghla Bikel (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5458896 Jan 19  2010 03 - Balam Ft. Julee - Bhalobashar Utshob (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5253112 Jan 19  2010 03 - Balam - Rimjhim (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8488651 Jan 20  2010 03 - Ovi - Doyal (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4098177 Jan 19  2010 04 - Balam Ft. Julee - Nagor Dola (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4777474 Jan 19  2010 04 - Balam - Oporupa (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 7859873 Jan 20  2010 04 - Zahid Pintu - Bhalo Bhalo Lage Na (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4649870 Jan 20  2010 05 - Balam Ft. Julee - Tobe Ki Hobe (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5164091 Jan 19  2010 05 - Balam - Korona Baron (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8317745 Jan 20  2010 05 - Julie - Golper Dinga (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5120240 Jan 20  2010 06 - Balam Ft. Julee - Bishoy (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4212820 Jan 19  2010 06 - Balam - Hridoyer Shirite (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 7751134 Jan 20  2010 06 - Zahid Pintu - Mon Majhi (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 9085594 Jan 20  2010 07 - Balam - Bhoboghure (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4713666 Jan 20  2010 07 - Balam Ft. Julee - Onuvhob (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 3885969 Jan 19  2010 07 - Balam - Matal Mon (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4745778 Jan 20  2010 08 - Balam Ft. Julee - Shuk Pakhi (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4701092 Jan 19  2010 08 - Balam - Pichutan (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 7099041 Jan 20  2010 08 - Ovi - Rang Dila (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5511855 Jan 20  2010 09 - Balam Ft. Julee - Eka (Instrumental) (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4539657 Jan 19  2010 09 - Balam - The Joker (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8725804 Jan 20  2010 09 - Ovi - Khacha (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8075902 Jan 20  2010 10 - Zahid Pintu - Bhurre Bhuriya Kuri (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4878225 Jan 20  2010 1 - Balam ft. Julee - Shonkhochil (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4998176 Jan 20  2010 2 - Balam ft. Julee - Tumihina (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5286156 Jan 20  2010 3 - Balam ft. Julee - Meghla Obhiman (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 3894353 Jan 20  2010 4 - Balam ft. Julee - Nesha Nesha Mon (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5100159 Jan 20  2010 5 - Balam ft. Julee - Icchegulo (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4310223 Jan 20  2010 6 - Balam ft. Julee - Hridoyer Thikana (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4550129 Jan 20  2010 7 - Balam ft. Julee - Dokkhina Duar (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4526298 Jan 20  2010 8 - Balam ft. Julee - Bolona (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4590256 Jan 20  2010 9 - Balam ft. Julee - Shopner Prithibi (music.com.bd).mp3
-r-xr-xr-x 1 palash palash    7987 Sep  2  2015 AlbumArtSmall.jpg
-r-xr-xr-x 1 palash palash 4558987 Jan 20  2010 Balam and Habib - Boro Eka (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4653550 Jan 20  2010 Balam and Habib - Dingulo (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5637849 Jan 20  2010 Balam and Habib - Ekhono Khuji (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5082483 Jan 20  2010 Balam and Habib - Firiye Dao (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5979525 Jan 20  2010 Balam and Habib - Hridoye (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 7003529 Jan 22  2010 Balam and Habib - Jhir Jhirey (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 6406365 Jan 22  2010 Balam and Habib - Merina (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5651953 Jan 22  2010 Balam and Habib - Onek Ratri (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4868799 Jan 22  2010 Balam and Habib - Onuvuti (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 8266818 Jan 22  2010 Balam and Habib - Prithibir Sob Sukh (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5556870 Jan 22  2010 Balam and Habib - Sundor Bikele (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5189059 Jan 22  2010 Balam and Habib - Tomari (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5069975 Jan 19  2010 Balam - Ashar Nichachol  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 3887909 Jan 19  2010 Balam - Biroher Sampan  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5175364 Jan 19  2010 Balam - Ek Mutho Roddur  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4631593 Jan 19  2010 Balam - Lukochuri  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4821766 Jan 19  2010 Balam - Nupur Baje  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 3408383 Jan 19  2010 Balam - Rondhonu  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 4952585 Jan 19  2010 Balam - Rupkotha  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 3455903 Jan 19  2010 Balam - Somorpon  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash 5110159 Jan 19  2010 Balam - Tomar Jonno  (music.com.bd).mp3
-r-xr-xr-x 1 palash palash   29378 Sep  2  2015 Folder.jpg
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $

```

See, with the help of Absolute and Relative Path we have checked what files or directories are present in different directories from our home directory. Not just one or two commands, with the help of absolute path and relative path we can run all commands in other directories. To know about the Linux / Unix system, it is recommended to have a clear idea of the file system. After having a solid foundation of Linux / Unix file system, we will move on to our next step. So, a thorough practice of file system navigation is strongly recommended. 

<div class='
