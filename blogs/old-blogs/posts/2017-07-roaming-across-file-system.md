---
title: "Roaming across the file system"
date: "2017-07-16T12:21:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045005)"
---

In our last discussion, we came to know how files and directories are present in the Unix/Linux file system and we also knew how to check which directory we are currently in. Stepping forward, we will see how we can navigate in any directory of the file system and how to check what all files are present in the current directory and what they are.
**From the previous post we learned that the Unix/Linux file system starts from the **root directory, here we will see how we can navigate the **root directory and what files or directories are present inside it. One thing to remember, since Unix/Linux is available in different flavors and the system administrator changes many configurations in each system, it may happen that two different systems can have different file system layout, consequently files and directories present in the root directory (or for that matter, within any directory) can be different. Maybe in my case, a particular file/directory is present but same is not present in your case, or vice-versa. 
**Let's see how we can navigate the root directory and find out the various files and directories present in it-

```
totan@Home-Computer ~ $ cd /
totan@Home-Computer / $

```

We can use the **cd (change directory) command to navigate from one directory to another directory. We know that in Unix/Linux system root directory is represented by **/, so when we run **cd / command, we will be able to navigate to the root directory. One more thing to note here, whenever we navigated to the root directory, our command prompt changed, earlier it was  totan@Home-Computer ~ $  now it has changed to  totan@Home-Computer / $. When we will move to another directories, we will understand this behaviour clearly. 

Let's see how we can find out what all files or directories are present in  the root directory. I have multiple Linux or Unix based systems and tried to check for each system what are the files/directories present in each of them- 

**Linux Mint

```
totan@Home-Computer / $ ls -l
total 136
drwxr-xr-x   2 root root 12288 Jul 30 21:08 bin
drwxr-xr-x   3 root root  4096 Sep  1 08:48 boot
drwxr-xr-x   2 root root  4096 Jan 10  2017 cdrom
drwxr-xr-x  22 root root  4340 Sep  1 08:40 dev
drwxr-xr-x 162 root root 12288 Sep  1 08:48 etc
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
dr-xr-xr-x 222 root root     0 Sep  1 08:39 proc
drwx------  19 root root  4096 Jul 29 00:01 root
drwxr-xr-x  33 root root  1120 Sep  1 08:47 run
drwxr-xr-x   2 root root 12288 Jul 30 21:08 sbin
drwxr-xr-x   2 root root  4096 Dec 14  2016 srv
dr-xr-xr-x  13 root root     0 Sep  1 08:39 sys
drwxrwxrwt  14 root root 20480 Sep  1 08:51 tmp
drwxr-xr-x   3 root root  4096 Jan 12  2017 u01
drwxr-xr-x  11 root root  4096 Jun 16 20:47 usr
drwxr-xr-x  11 root root  4096 Dec 14  2016 var
lrwxrwxrwx   1 root root    29 Sep  1 08:47 vmlinuz -> boot/vmlinuz-4.4.0-93-generic
lrwxrwxrwx   1 root root    29 Aug 20 12:11 vmlinuz.old -> boot/vmlinuz-4.4.0-92-generic
totan@Home-Computer / $

```

**Oracle Linux

```
[root@docker /]#ls -l
total 14296
lrwxrwxrwx.   1 root root          7 Jul 14  2016 bin -> usr/bin
dr-xr-xr-x.   3 root root       4096 Jul 14  2016 boot
drwxr-xr-x   19 root root       3180 Sep  1 09:08 dev
drwxr-xr-x.   1 root root       4508 Sep  1 09:09 etc
-rw-r--r--    1 root palash 14614796 Aug 21 13:25 file_tree
drwxr-xr-x.   1 root root         12 Aug 11 09:40 home
lrwxrwxrwx.   1 root root          7 Jul 14  2016 lib -> usr/lib
lrwxrwxrwx.   1 root root          9 Jul 14  2016 lib64 -> usr/lib64
drwxr-xr-x.   1 root root         16 Aug 24 10:02 media
drwxr-xr-x.   1 root root          0 Sep 11  2015 mnt
drwxr-xr-x.   1 root root         52 Jul 14  2016 opt
dr-xr-xr-x  133 root root          0 Sep  1 09:08 proc
dr-xr-x---.   1 root palash      260 Aug 24 10:00 root
drwxr-xr-x   32 root root       1080 Sep  1 09:09 run
lrwxrwxrwx.   1 root root          8 Jul 14  2016 sbin -> usr/sbin
drwxr-xr-x.   1 root root          0 Sep 11  2015 srv
dr-xr-xr-x   13 root root          0 Sep  1 09:08 sys
drwxrwxrwt.   1 root root        586 Sep  1 09:16 tmp
drwxr-xr-x.   1 root root        106 Jul 14  2016 usr
drwxr-xr-x.   1 root root        200 Sep  1 09:08 var

```

**FreeBSD

![image](https://1.bp.blogspot.com/-seRN2dIX3CQ/WajZXxIvxNI/AAAAAAAAHeY/1UGXwFKdo7kCmcU3OH2cVgU85yfnKGCowCEwYBhgL/s1600/Screenshot%2Bfrom%2B2017-09-01%2B09-20-54.png)

We now know what all files/directories are present in the root directory of different systems, by running the **ls -l command, but how do we understand which one is directory and which one is file. Check all outputs once again and note that In some cases, some lines start with the letter **d, sometimes with **l and sometimes with **- and the last column of the output is the name of a file/directory.Soon we'll check these outputs and will know what does it signify while learning **ls command in detail. But for now, just knowing that the lines which start with the letter **d is a directory and if it starts with **- then it is file.**Now we know what is in **root directory and we will try to find out what child directories have within them. Again I want to remind you that the file system may not be the same for everyone, so the next commands that I am running on my file system, you can change according to your file system -

```
totan@Home-Computer / $ cd /
totan@Home-Computer / $ pwd
/
totan@Home-Computer / $ ls -l
total 136
drwxr-xr-x   2 root root 12288 Jul 30 21:08 bin
drwxr-xr-x   3 root root  4096 Sep  1 08:48 boot
drwxr-xr-x   2 root root  4096 Jan 10  2017 cdrom
drwxr-xr-x  22 root root  4340 Sep  1 08:40 dev
drwxr-xr-x 162 root root 12288 Sep  1 08:48 etc
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
dr-xr-xr-x 230 root root     0 Sep  1 08:39 proc
drwx------  19 root root  4096 Jul 29 00:01 root
drwxr-xr-x  33 root root  1120 Sep  1 08:47 run
drwxr-xr-x   2 root root 12288 Jul 30 21:08 sbin
drwxr-xr-x   2 root root  4096 Dec 14  2016 srv
dr-xr-xr-x  13 root root     0 Sep  1 09:07 sys
drwxrwxrwt  15 root root 20480 Sep  1 09:39 tmp
drwxr-xr-x   3 root root  4096 Jan 12  2017 u01
drwxr-xr-x  11 root root  4096 Jun 16 20:47 usr
drwxr-xr-x  11 root root  4096 Dec 14  2016 var
lrwxrwxrwx   1 root root    29 Sep  1 08:47 vmlinuz -> boot/vmlinuz-4.4.0-93-generic
lrwxrwxrwx   1 root root    29 Aug 20 12:11 vmlinuz.old -> boot/vmlinuz-4.4.0-92-generic
totan@Home-Computer / $ cd bin
totan@Home-Computer /bin $ pwd
/bin
totan@Home-Computer /bin $ ls -l
total 17456
-rwxr-xr-x 1 root root    6456 Jul  4 20:24 archdetect
-rwxr-xr-x 1 root root 1037440 May 26 17:52 bash
-rwxr-xr-x 1 root root  520992 Jun 16 05:16 btrfs
-rwxr-xr-x 1 root root  249464 Jun 16 05:16 btrfs-calc-size
lrwxrwxrwx 1 root root       5 Jun 16 05:16 btrfsck -> btrfs
-rwxr-xr-x 1 root root  278376 Jun 16 05:16 btrfs-convert
-rwxr-xr-x 1 root root  249464 Jun 16 05:16 btrfs-debug-tree
-rwxr-xr-x 1 root root  245368 Jun 16 05:16 btrfs-find-root
-rwxr-xr-x 1 root root  270136 Jun 16 05:16 btrfs-image
-rwxr-xr-x 1 root root  249464 Jun 16 05:16 btrfs-map-logical
-rwxr-xr-x 1 root root  245368 Jun 16 05:16 btrfs-select-super
-rwxr-xr-x 1 root root  253816 Jun 16 05:16 btrfs-show-super
-rwxr-xr-x 1 root root  249464 Jun 16 05:16 btrfstune
-rwxr-xr-x 1 root root  245368 Jun 16 05:16 btrfs-zero-log
-rwxr-xr-x 1 root root   31288 May 20  2015 bunzip2
-rwxr-xr-x 1 root root 1964536 Aug 19  2015 busybox
-rwxr-xr-x 1 root root   31288 May 20  2015 bzcat
lrwxrwxrwx 1 root root       6 Jan 10  2017 bzcmp -> bzdiff
-rwxr-xr-x 1 root root    2140 May 20  2015 bzdiff
lrwxrwxrwx 1 root root       6 Jan 10  2017 bzegrep -> bzgrep
-rwxr-xr-x 1 root root    4877 May 20  2015 bzexe
lrwxrwxrwx 1 root root       6 Jan 10  2017 bzfgrep -> bzgrep
-rwxr-xr-x 1 root root    3642 May 20  2015 bzgrep
-rwxr-xr-x 1 root root   31288 May 20  2015 bzip2
-rwxr-xr-x 1 root root   14616 May 20  2015 bzip2recover
lrwxrwxrwx 1 root root       6 Jan 10  2017 bzless -> bzmore
-rwxr-xr-x 1 root root    1297 May 20  2015 bzmore
-rwxr-xr-x 1 root root   52080 Mar  2 23:37 cat
-rwxr-xr-x 1 root root   14752 Feb  7  2016 chacl
-rwxr-xr-x 1 root root   60272 Mar  2 23:37 chgrp
-rwxr-xr-x 1 root root   56112 Mar  2 23:37 chmod
-rwxr-xr-x 1 root root   64368 Mar  2 23:37 chown
-rwxr-xr-x 1 root root   10536 Sep 22  2016 chvt
-rwxr-xr-x 1 root root  151024 Mar  2 23:37 cp
-rwxr-xr-x 1 root root  141472 Feb 18  2016 cpio
lrwxrwxrwx 1 root root      21 Jun  9 23:56 csh -> /etc/alternatives/csh
-rwxr-xr-x 1 root root  154072 Feb 18  2016 dash
-rwxr-xr-x 1 root root   68464 Mar  2 23:37 date
-rwxr-xr-x 1 root root   72632 Mar  2 23:37 dd
-rwxr-xr-x 1 root root   97912 Mar  2 23:37 df
-rwxr-xr-x 1 root root  126584 Mar  2 23:37 dir
-rwxr-xr-x 1 root root   60680 Jun 15 03:21 dmesg
lrwxrwxrwx 1 root root       8 Jan 10  2017 dnsdomainname -> hostname
lrwxrwxrwx 1 root root       8 Jan 10  2017 domainname -> hostname
-rwxr-xr-x 1 root root   82328 Sep 22  2016 dumpkeys
-rwxr-xr-x 1 root root   31376 Mar  2 23:37 echo
-rwxr-xr-x 1 root root   51512 Apr 26  2014 ed
-rwxr-xr-x 1 root root   36504 Jul 14  2015 efibootmgr
-rwxr-xr-x 1 root root      28 Apr 29  2016 egrep
-rwxr-xr-x 1 root root   27280 Mar  2 23:37 false
-rwxr-xr-x 1 root root   10544 Sep 22  2016 fgconsole
-rwxr-xr-x 1 root root      28 Apr 29  2016 fgrep
-rwxr-xr-x 1 root root   49576 Jun 15 03:21 findmnt
-rwxr-xr-x 1 root root    1185 Jun 16 05:16 fsck.btrfs
-rwxr-xr-x 1 root root   36024 Feb  7  2016 fuser
-rwsr-xr-x 1 root root   30800 Jul 12  2016 fusermount
-rwxr-xr-x 1 root root   23752 Feb  7  2016 getfacl
-rwxr-xr-x 1 root root  211224 Apr 29  2016 grep
-rwxr-xr-x 1 root root    2301 Oct 27  2014 gunzip
-rwxr-xr-x 1 root root    5927 Oct 27  2014 gzexe
-rwxr-xr-x 1 root root   98240 Oct 27  2014 gzip
-rwxr-xr-x 1 root root  192760 Mar  1  2016 hciconfig
-rwxr-xr-x 1 root root   14800 Nov 24  2015 hostname
-rwxr-xr-x 1 root root  376192 May  5 21:54 ip
-rwxr-xr-x 1 root root  498936 Jul 19 05:26 journalctl
-rwxr-xr-x 1 root root   10536 Sep 22  2016 kbd_mode
-rwxr-xr-x 1 root root   30792 Dec 10  2015 keyctl
-rwxr-xr-x 1 root root   23152 Nov 21  2016 kill
-rwxr-xr-x 1 root root  150600 Jul 25 20:05 kmod
lrwxrwxrwx 1 root root      21 Jun  9 23:58 ksh -> /etc/alternatives/ksh
-rwxr-xr-x 1 root root  170728 Apr  5 18:39 less
-rwxr-xr-x 1 root root   10256 Apr  5 18:39 lessecho
lrwxrwxrwx 1 root root       8 Apr  5 18:39 lessfile -> lesspipe
-rwxr-xr-x 1 root root   19824 Apr  5 18:39 lesskey
-rwxr-xr-x 1 root root    7764 Apr  5 18:39 lesspipe
-rwxr-xr-x 1 root root  156984 Apr 14  2016 lksh
-rwxr-xr-x 1 root root   56152 Mar  2 23:37 ln
-rwxr-xr-x 1 root root  111496 Sep 22  2016 loadkeys
-rwxr-xr-x 1 root root   48128 May 17 05:07 login
-rwxr-xr-x 1 root root  453848 Jul 19 05:26 loginctl
-rwxr-xr-x 1 root root  105136 Jan 28  2017 lowntfs-3g
-rwxr-xr-x 1 root root  126584 Mar  2 23:37 ls
-rwxr-xr-x 1 root root   77280 Jun 15 03:21 lsblk
lrwxrwxrwx 1 root root       4 Jul 25 20:05 lsmod -> kmod
-rwxr-xr-x 1 root root   76848 Mar  2 23:37 mkdir
-rwxr-xr-x 1 root root  265928 Jun 16 05:16 mkfs.btrfs
-rwxr-xr-x 1 root root   64496 Mar  2 23:37 mknod
-rwxr-xr-x 1 root root  269088 Apr 14  2016 mksh
lrwxrwxrwx 1 root root      23 Apr 14  2016 mksh-static -> /usr/lib/klibc/bin/mksh
-rwxr-xr-x 1 root root   39728 Mar  2 23:37 mktemp
-rwxr-xr-x 1 root root   39768 Jun 15 03:21 more
-rwsr-xr-x 1 root root   40152 Jun 15 03:21 mount
-rwxr-xr-x 1 root root   14768 Jun 15 03:21 mountpoint
lrwxrwxrwx 1 root root      20 Jan 10  2017 mt -> /etc/alternatives/mt
-rwxr-xr-x 1 root root   68824 Feb 18  2016 mt-gnu
-rwxr-xr-x 1 root root  130488 Mar  2 23:37 mv
-rwxr-xr-x 1 root root  208480 Feb 15  2017 nano
lrwxrwxrwx 1 root root      20 Jan 10  2017 nc -> /etc/alternatives/nc
-rwxr-xr-x 1 root root   31248 Dec  4  2012 nc.openbsd
lrwxrwxrwx 1 root root      24 Jan 10  2017 netcat -> /etc/alternatives/netcat
-rwxr-xr-x 1 root root  119624 Jun 30  2014 netstat
-rwxr-xr-x 1 root root  678496 Jul 19 05:26 networkctl
lrwxrwxrwx 1 root root       8 Jan 10  2017 nisdomainname -> hostname
-rwsr-xr-x 1 root root  142032 Jan 28  2017 ntfs-3g
-rwxr-xr-x 1 root root   10312 Jan 28  2017 ntfs-3g.probe
-rwxr-xr-x 1 root root   67608 Jan 28  2017 ntfs-3g.secaudit
-rwxr-xr-x 1 root root   18424 Jan 28  2017 ntfs-3g.usermap
-rwxr-xr-x 1 root root   26728 Jan 28  2017 ntfscat
-rwxr-xr-x 1 root root   30824 Jan 28  2017 ntfscluster
-rwxr-xr-x 1 root root   34920 Jan 28  2017 ntfscmp
-rwxr-xr-x 1 root root   34928 Jan 28  2017 ntfsfallocate
-rwxr-xr-x 1 root root   39024 Jan 28  2017 ntfsfix
-rwxr-xr-x 1 root root   55416 Jan 28  2017 ntfsinfo
-rwxr-xr-x 1 root root   31928 Jan 28  2017 ntfsls
-rwxr-xr-x 1 root root   30824 Jan 28  2017 ntfsmove
-rwxr-xr-x 1 root root   38944 Jan 28  2017 ntfstruncate
-rwxr-xr-x 1 root root   47752 Jan 28  2017 ntfswipe
lrwxrwxrwx 1 root root       6 Jan 10  2017 open -> openvt
-rwxr-xr-x 1 root root   18968 Sep 22  2016 openvt
lrwxrwxrwx 1 root root      14 Jan 10  2017 pidof -> /sbin/killall5
-rwsr-xr-x 1 root root   44168 May  8  2014 ping
-rwsr-xr-x 1 root root   44680 May  8  2014 ping6
-rwxr-xr-x 1 root root   35504 Sep  4  2016 plymouth
-rwxr-xr-x 1 root root   97408 Nov 21  2016 ps
-rwxr-xr-x 1 root root   31472 Mar  2 23:37 pwd
lrwxrwxrwx 1 root root       4 May 26 17:52 rbash -> bash
-rwxr-xr-x 1 root root   39632 Mar  2 23:37 readlink
-rwxr-xr-x 1 root root      89 Apr 26  2014 red
-rwxr-xr-x 1 root root   60272 Mar  2 23:37 rm
-rwxr-xr-x 1 root root   39632 Mar  2 23:37 rmdir
lrwxrwxrwx 1 root root       4 Feb 15  2017 rnano -> nano
-rwxr-xr-x 1 root root   19320 Jan 26  2016 run-parts
lrwxrwxrwx 1 root root      22 Jun 10 00:01 rzsh -> /etc/alternatives/rzsh
-rwxr-xr-x 1 root root   73424 Feb 12  2016 sed
-rwxr-xr-x 1 root root   36296 Feb  7  2016 setfacl
-rwxr-xr-x 1 root root   39952 Sep 22  2016 setfont
-rwxr-xr-x 1 root root   30875 Feb  1  2017 setupcon
lrwxrwxrwx 1 root root       4 Jan 10  2017 sh -> dash
lrwxrwxrwx 1 root root       4 Jan 10  2017 sh.distrib -> dash
-rwxr-xr-x 1 root root   31408 Mar  2 23:37 sleep
-rwxr-xr-x 1 root root  115816 May  5 21:54 ss
lrwxrwxrwx 1 root root       7 Jan 10  2017 static-sh -> busybox
-rwxr-xr-x 1 root root   72496 Mar  2 23:37 stty
-rwsr-xr-x 1 root root   40128 May 17 05:07 su
-rwxr-xr-x 1 root root   31408 Mar  2 23:37 sync
-rwxr-xr-x 1 root root  659848 Jul 19 05:26 systemctl
lrwxrwxrwx 1 root root      20 Jul 19 05:26 systemd -> /lib/systemd/systemd
-rwxr-xr-x 1 root root   51656 Jul 19 05:26 systemd-ask-password
-rwxr-xr-x 1 root root   39344 Jul 19 05:26 systemd-escape
-rwxr-xr-x 1 root root   64080 Jul 19 05:27 systemd-hwdb
-rwxr-xr-x 1 root root  281840 Jul 19 05:26 systemd-inhibit
-rwxr-xr-x 1 root root   47544 Jul 19 05:26 systemd-machine-id-setup
-rwxr-xr-x 1 root root   35248 Jul 19 05:26 systemd-notify
-rwxr-xr-x 1 root root  133704 Jul 19 05:26 systemd-tmpfiles
-rwxr-xr-x 1 root root   68032 Jul 19 05:26 systemd-tty-ask-password-agent
-rwxr-xr-x 1 root root   23144 Jun 15 03:21 tailf
-rwxr-xr-x 1 root root  383632 Nov 17  2016 tar
lrwxrwxrwx 1 root root      13 Jun  9 23:56 tcsh -> /usr/bin/tcsh
-rwxr-xr-x 1 root root   10416 Jan 26  2016 tempfile
-rwxr-xr-x 1 root root   64432 Mar  2 23:37 touch
-rwxr-xr-x 1 root root   27280 Mar  2 23:37 true
-rwxr-xr-x 1 root root  449136 Jul 19 05:27 udevadm
-rwxr-xr-x 1 root root   14328 Jul 12  2016 ulockmgr_server
-rwsr-xr-x 1 root root   27608 Jun 15 03:21 umount
-rwxr-xr-x 1 root root   31440 Mar  2 23:37 uname
-rwxr-xr-x 1 root root    2301 Oct 27  2014 uncompress
-rwxr-xr-x 1 root root    2762 Sep 22  2016 unicode_start
-rwxr-xr-x 1 root root  126584 Mar  2 23:37 vdir
-rwxr-xr-x 1 root root   10552 Mar  3  2016 vmmouse_detect
-rwxr-xr-x 1 root root   31376 Jun 15 03:21 wdctl
-rwxr-xr-x 1 root root     946 Jan 26  2016 which
-rwxr-xr-x 1 root root   27432 Jan 19  2016 whiptail
lrwxrwxrwx 1 root root       8 Jan 10  2017 ypdomainname -> hostname
-rwxr-xr-x 1 root root    1937 Oct 27  2014 zcat
-rwxr-xr-x 1 root root    1777 Oct 27  2014 zcmp
-rwxr-xr-x 1 root root    5764 Oct 27  2014 zdiff
-rwxr-xr-x 1 root root     140 Oct 27  2014 zegrep
-rwxr-xr-x 1 root root     140 Oct 27  2014 zfgrep
-rwxr-xr-x 1 root root    2131 Oct 27  2014 zforce
-rwxr-xr-x 1 root root    5938 Oct 27  2014 zgrep
-rwxr-xr-x 1 root root    2037 Oct 27  2014 zless
-rwxr-xr-x 1 root root    1910 Oct 27  2014 zmore
-rwxr-xr-x 1 root root    5047 Oct 27  2014 znew
lrwxrwxrwx 1 root root      21 Jun 10 00:01 zsh -> /etc/alternatives/zsh
-rwxr-xr-x 1 root root  771552 Feb  7  2016 zsh5
totan@Home-Computer /bin $ cd /
totan@Home-Computer / $ pwd
/
totan@Home-Computer / $ ls -l
total 136
drwxr-xr-x   2 root root 12288 Jul 30 21:08 bin
drwxr-xr-x   3 root root  4096 Sep  1 08:48 boot
drwxr-xr-x   2 root root  4096 Jan 10  2017 cdrom
drwxr-xr-x  22 root root  4340 Sep  1 08:40 dev
drwxr-xr-x 162 root root 12288 Sep  1 08:48 etc
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
dr-xr-xr-x 231 root root     0 Sep  1 08:39 proc
drwx------  19 root root  4096 Jul 29 00:01 root
drwxr-xr-x  33 root root  1120 Sep  1 08:47 run
drwxr-xr-x   2 root root 12288 Jul 30 21:08 sbin
drwxr-xr-x   2 root root  4096 Dec 14  2016 srv
dr-xr-xr-x  13 root root     0 Sep  1 09:07 sys
drwxrwxrwt  15 root root 20480 Sep  1 09:39 tmp
drwxr-xr-x   3 root root  4096 Jan 12  2017 u01
drwxr-xr-x  11 root root  4096 Jun 16 20:47 usr
drwxr-xr-x  11 root root  4096 Dec 14  2016 var
lrwxrwxrwx   1 root root    29 Sep  1 08:47 vmlinuz -> boot/vmlinuz-4.4.0-93-generic
lrwxrwxrwx   1 root root    29 Aug 20 12:11 vmlinuz.old -> boot/vmlinuz-4.4.0-92-generic
totan@Home-Computer / $ cd boot
totan@Home-Computer /boot $ pwd
/boot
totan@Home-Computer /boot $ ls -l
total 779672
-rw-r--r-- 1 root root  1243479 Dec  3  2016 abi-4.4.0-53-generic
-rw-r--r-- 1 root root  1244118 Jan  7  2017 abi-4.4.0-59-generic
-rw-r--r-- 1 root root  1244118 Jan 18  2017 abi-4.4.0-62-generic
-rw-r--r-- 1 root root  1245512 Feb  2  2017 abi-4.4.0-63-generic
-rw-r--r-- 1 root root  1245512 Feb 20  2017 abi-4.4.0-64-generic
-rw-r--r-- 1 root root  1245512 Mar  3 23:55 abi-4.4.0-66-generic
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
-rw-r--r-- 1 root root   190247 Mar  3 23:55 config-4.4.0-66-generic
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
-rw------- 1 root root  3883990 Mar  3 23:55 System.map-4.4.0-66-generic
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
-rw------- 1 root root  7087024 Mar  3 23:55 vmlinuz-4.4.0-66-generic
-rw------- 1 root root  7089552 Apr 27 23:54 vmlinuz-4.4.0-78-generic
-rw------- 1 root root  7091696 May 18 03:39 vmlinuz-4.4.0-79-generic
-rw------- 1 root root  7092784 Jun 14 17:54 vmlinuz-4.4.0-81-generic
-rw------- 1 root root  7092720 Jun 27 01:15 vmlinuz-4.4.0-83-generic
-rw------- 1 root root  7095888 Jul 18 20:30 vmlinuz-4.4.0-87-generic
-rw------- 1 root root  7098096 Aug  1 03:55 vmlinuz-4.4.0-89-generic
-rw------- 1 root root  7097936 Aug  8 19:28 vmlinuz-4.4.0-91-generic
-rw------- 1 root root  7098032 Aug 10 16:32 vmlinuz-4.4.0-92-generic
-rw------- 1 root root  7097296 Aug 12 05:10 vmlinuz-4.4.0-93-generic
totan@Home-Computer /boot $

```

This is how we can navigate to different directories. Now seeing all the outputs, we can understand that when we are navigating from one directory to another, system shows the name of the current directory **(pwd) in the prompt, which helps us to remember which directory we are working in.

This way we can navigate to different directories of the file system but there is a problem, we have to run the **cd / command again and again to go back to the root directory and from there we have to navigate to its child directories. While there is no harm in it but it is cumbersome at times. So we'll learn few shortcuts now, which are very useful when navigating Unix or Linux file system. 

After navigating many directories we often get lost and there is a shortcut to return to our **home, **cd ~ helps us to return to our **home directory directly -

```
totan@Home-Computer /boot $ pwd
/boot
totan@Home-Computer /boot $ cd ~
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $

```

Similarly, if you have to go back to the directory you were just right before (in my case **/boot ), **cd - will help you -

**

```
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ cd -
/boot
totan@Home-Computer /boot $ pwd
/boot

```

If you have to navigate to the parent directory of the current directory, you will use **cd ..

```
totan@Home-Computer /boot $ pwd
/boot
totan@Home-Computer /boot $ cd ..
totan@Home-Computer / $ pwd
/
totan@Home-Computer / $ cd bin
totan@Home-Computer /bin $ pwd
/bin
totan@Home-Computer /bin $ cd ..
totan@Home-Computer / $ cd mnt
totan@Home-Computer /mnt $ pwd
/mnt
totan@Home-Computer /mnt $ cd ..
totan@Home-Computer / $ cd ~
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $ cd ..
totan@Home-Computer /home $ pwd
/home
totan@Home-Computer /home $ ls -l
total 8
drwxr-xr-x 55 palash palash 4096 Aug 25 21:57 palash
drwxr-xr-x 46 totan  totan  4096 Sep  1 09:41 totan
totan@Home-Computer /home $ cd palash
totan@Home-Computer /home/palash $ pwd
/home/palash
totan@Home-Computer /home/palash $ cd ..
totan@Home-Computer /home $ pwd
/home
totan@Home-Computer /home $ cd totan
totan@Home-Computer ~ $ pwd
/home/totan
totan@Home-Computer ~ $

```

We can navigate the file system easily with these shortcuts. Now it's your turn to use the commands and the shortcuts to navigate around the file system and get an idea of how the file system looks like.

In our next post, we will learn more shortcuts and file related commands.

<div class='
