# লিনাক্সের ভিত


আমাদের পূর্ববর্তী পোস্টটি থেকে আমরা জানতে পারি যে, গ্নু/লিনাক্স একটি অপারেটিং সিস্টেম যার সূচনা করেছিলেন লিনাস টরভাল্ডস ১৯৯১ সালে এবং সেটি সময়ের সাথে বিস্তারিত হতে হতে আজ সর্বব্যাপী হয়ে উঠেছে।

এবার আমরা দেখে নেবো লিনাক্সের বিল্ডিং ব্লক গুলি কি কি। যে কোনো অপারেটিং সিস্টেম অনেকগুলি প্রোগ্রামের সমন্বয়ে সৃষ্ট একটি সিস্টেম। লিনাক্সও তার ব্যতিক্রম নয়। যে কোনো লিনাক্স সিস্টেম চলার জন্যে সর্বনিম্ন কিছু প্রোগ্রাম অবশ্যই লাগে, তার মধ্যে অন্যতম প্রধান কয়েকটি সম্পর্কে জানার চেষ্টা করবো -
**কার্নেল - এটি হলো যেকোনো লিনাক্স সিস্টেমের সর্বনিম্ন প্রোগ্রাম যেটি কিনা সরাসরি হার্ডওয়ারের সাথে যোগাযোগ করে। এক কথায় বলা চলে যে কোনো লিনাক্স অপারেটিং সিস্টেমের প্রাণ ভোমরা হলো কার্নেল। আসলে লিনাক্স নামের কোনো অপারেটিং সিস্টেমই নেই, যেকোনো লিনাক্স (বা সঠিকভাবে বলতে গেলে গ্নু/লিনাক্স) অপারেটিং সিস্টেম হচ্ছে আসলে লিনাক্স কার্নেল এবং কিছু ওপেনসোর্স টুলসের সমন্বয়। ১৯৯১ সালে লিনাস টরভাল্ডস আসলে এই লিনাক্স কর্নেলের গোঁড়াপত্তন করেন এবং তার সাথে গ্নু টুলসের সমন্বয় ঘটিয়ে প্রথম লিনাক্স অপারেটিং সিস্টেম (আবারও, সঠিকভাবে বলতে গেলে লিনাক্স বেসড অপারেটিং সিস্টেম) বের করেন।
**ফাইল এবং প্রসেস - যেকোনো লিনাক্স বেসড অপারেটিং সিস্টেম আসলে কিছু ফাইল এবং প্রসেসের সমন্বয়ে গড়ে ওঠে। ফাইল বলতে যে কোনো ধরণের ফাইল হতে পারে - মিডিয়া ফাইল, প্রোগ্রাম ফাইলস, কন্ট্রোল ফাইলস ইত্যাদি। সাধারণত ফাইল গুলি সাজানো থাকে বিভিন্ন ডিরেক্টরির মধ্যে। লিনাক্সে সাধারণত বেশির ভাগ ফাইলই হয় টেক্সট বেসড অর্থাৎ কিনা আপনি যেকোনো ফাইলকে যেকোনো সাধারণ এডিটরে ওপেন করে পড়তে এবং বদলাতে পারেন। এখানেই আছে লিনাক্সের সৌন্দর্য্য, আপনি চাইলেই যেকোনো ফাইল ওপেন করে পড়ে বুঝতে পারেন আপনার কম্পিউটার কি কাজ করছে বা চাইলে আপনি সেটা বদলে আপনার নিজের মতো চালনা করতে পারেন এবং শুধু তাইই না, আপনার মন হলে আপনার পরিবর্তিত সংস্করণ আপনি চাইলে সকলের মধ্যে বিতরণ করতে পারেন বা সেই পরিবর্তিত সংস্করণের জন্যে আপনি মূল্য ধার্য করতে পারেন। এটা সত্যিই একটা বিশাল স্বাধীনতা।

প্রসেস বলতে লিনাক্সে বোঝানো হয়, সেই প্রোগ্রাম যেটি হার্ডওয়ারের মাধ্যমে সঞ্চালিত হচ্ছে। অর্থাৎ, আপনি যেমন বর্তমানে এই পোস্টটি পড়ছেন কোনো একটা ব্রাউজারে, সেক্ষেত্রে ব্রাউজার বর্তমানে আপনার সিস্টেমে চলমান প্রসেস। সাধারণত প্রসেস গুলি ফাইলে স্টোর করা থাকে এবং ইউজারের রিকোয়েস্ট অনুসারে সেগুলিকে ফাইল থেকে মেমোরিতে (RAM) লোড করে হার্ডওয়ারের মাধ্যমে সঞ্চালনা করা হয়। সুতরাং বুঝতেই পারছেন, একটি চলমান লিনাক্স বেসড অপারেটিং সিস্টেমে যেকোনো সময়ে কিছু না কিছু ফাইল অবশ্যই আছে এবং কোনো না কোনো প্রসেস অবশ্যই চলছে।
**ডিরেক্টরি - ডিরেক্টরি বলতে বোঝায় কিছু ফাইল বা ডিরেক্টরির সমন্বয়। অর্থাৎ, একটি ডিরেক্টরির মধ্যে এক বা একাধিক ফাইল এবং/অথবা এক বা একাধিক ডিরেক্টরি থাকতে পারে এবং আপনি যেকোনো সময়ে কোনো না কোনো ডিরেক্টরির মধ্যে নিশ্চয় অবস্থান করছেন। যেকোনো লিনাক্স বেসড অপারেটিং সিস্টেমের ডিরেক্টরি শুরু হয় **root (/) থেকে। উদাহরণস্বরূপ, এই মুহূর্তে আমার অপারেটিং সিস্টেমের ডিরেক্টরি স্ট্রাকচার খানিকটা এই রকম -

```
.
├── bin
│   ├── archdetect
│   ├── bash
│   ├── btrfs
│   ├── btrfs-calc-size
│   ├── btrfsck -> btrfs
│   ├── btrfs-convert
│   ├── btrfs-debug-tree
│   ├── btrfs-find-root
│   ├── btrfs-image
│   ├── btrfs-map-logical
│   ├── btrfs-select-super
│   ├── btrfs-show-super
│   ├── btrfstune
│   ├── btrfs-zero-log
│   ├── bunzip2
│   ├── busybox
│   ├── bzcat
│   ├── bzcmp -> bzdiff
│   ├── bzdiff
│   ├── bzegrep -> bzgrep
│   ├── bzexe
│   ├── bzfgrep -> bzgrep
│   ├── bzgrep
│   ├── bzip2
│   ├── bzip2recover
│   ├── bzless -> bzmore
│   ├── bzmore
│   ├── cat
│   ├── chacl
│   ├── chgrp
│   ├── chmod
│   ├── chown
│   ├── chvt
│   ├── cp
│   ├── cpio
│   ├── dash
│   ├── date
│   ├── dd
│   ├── df
│   ├── dir
│   ├── dmesg
│   ├── dnsdomainname -> hostname
│   ├── domainname -> hostname
│   ├── dumpkeys
│   ├── echo
│   ├── ed
│   ├── efibootmgr
│   ├── egrep
│   ├── false
│   ├── fgconsole
│   ├── fgrep
│   ├── findmnt
│   ├── fsck.btrfs
│   ├── fuser
│   ├── fusermount
│   ├── getfacl
│   ├── grep
│   ├── gunzip
│   ├── gzexe
│   ├── gzip
│   ├── hciconfig
│   ├── hostname
│   ├── ip
│   ├── journalctl
│   ├── kbd_mode
│   ├── keyctl
│   ├── kill
│   ├── kmod
│   ├── less
│   ├── lessecho
│   ├── lessfile -> lesspipe
│   ├── lesskey
│   ├── lesspipe
│   ├── ln
│   ├── loadkeys
│   ├── login
│   ├── loginctl
│   ├── lowntfs-3g
│   ├── ls
│   ├── lsblk
│   ├── lsmod -> kmod
│   ├── mkdir
│   ├── mkfs.btrfs
│   ├── mknod
│   ├── mktemp
│   ├── more
│   ├── mount
│   ├── mountpoint
│   ├── mt -> /etc/alternatives/mt
│   ├── mt-gnu
│   ├── mv
│   ├── nano
│   ├── nc -> /etc/alternatives/nc
│   ├── nc.openbsd
│   ├── netcat -> /etc/alternatives/netcat
│   ├── netstat
│   ├── networkctl
│   ├── nisdomainname -> hostname
│   ├── ntfs-3g
│   ├── ntfs-3g.probe
│   ├── ntfs-3g.secaudit
│   ├── ntfs-3g.usermap
│   ├── ntfscat
│   ├── ntfscluster
│   ├── ntfscmp
│   ├── ntfsfallocate
│   ├── ntfsfix
│   ├── ntfsinfo
│   ├── ntfsls
│   ├── ntfsmove
│   ├── ntfstruncate
│   ├── ntfswipe
│   ├── open -> openvt
│   ├── openvt
│   ├── pidof -> /sbin/killall5
│   ├── ping
│   ├── ping6
│   ├── plymouth
│   ├── ps
│   ├── pwd
│   ├── rbash -> bash
│   ├── readlink
│   ├── red
│   ├── rm
│   ├── rmdir
│   ├── rnano -> nano
│   ├── run-parts
│   ├── sed
│   ├── setfacl
│   ├── setfont
│   ├── setupcon
│   ├── sh -> dash
│   ├── sh.distrib -> dash
│   ├── sleep
│   ├── ss
│   ├── static-sh -> busybox
│   ├── stty
│   ├── su
│   ├── sync
│   ├── systemctl
│   ├── systemd -> /lib/systemd/systemd
│   ├── systemd-ask-password
│   ├── systemd-escape
│   ├── systemd-hwdb
│   ├── systemd-inhibit
│   ├── systemd-machine-id-setup
│   ├── systemd-notify
│   ├── systemd-tmpfiles
│   ├── systemd-tty-ask-password-agent
│   ├── tailf
│   ├── tar
│   ├── tempfile
│   ├── touch
│   ├── true
│   ├── udevadm
│   ├── ulockmgr_server
│   ├── umount
│   ├── uname
│   ├── uncompress
│   ├── unicode_start
│   ├── vdir
│   ├── vmmouse_detect
│   ├── wdctl
│   ├── which
│   ├── whiptail
│   ├── ypdomainname -> hostname
│   ├── zcat
│   ├── zcmp
│   ├── zdiff
│   ├── zegrep
│   ├── zfgrep
│   ├── zforce
│   ├── zgrep
│   ├── zless
│   ├── zmore
│   └── znew
├── boot
│   ├── abi-4.4.0-53-generic
│   ├── abi-4.4.0-59-generic
│   ├── abi-4.4.0-62-generic
│   ├── abi-4.4.0-63-generic
│   ├── abi-4.4.0-64-generic
│   ├── abi-4.4.0-66-generic
│   ├── abi-4.4.0-78-generic
│   ├── config-4.4.0-53-generic
│   ├── config-4.4.0-59-generic
│   ├── config-4.4.0-62-generic
│   ├── config-4.4.0-63-generic
│   ├── config-4.4.0-64-generic
│   ├── config-4.4.0-66-generic
│   ├── config-4.4.0-78-generic
│   ├── grub
│   │   ├── fonts
│   │   ├── gfxblacklist.txt
│   │   ├── grub.cfg
│   │   ├── grubenv
│   │   ├── i386-pc
│   │   ├── locale
│   │   └── unicode.pf2
│   ├── initrd.img-4.4.0-53-generic
│   ├── initrd.img-4.4.0-59-generic
│   ├── initrd.img-4.4.0-62-generic
│   ├── initrd.img-4.4.0-63-generic
│   ├── initrd.img-4.4.0-64-generic
│   ├── initrd.img-4.4.0-66-generic
│   ├── initrd.img-4.4.0-78-generic
│   ├── memtest86+.bin
│   ├── memtest86+.elf
│   ├── memtest86+_multiboot.bin
│   ├── System.map-4.4.0-53-generic
│   ├── System.map-4.4.0-59-generic
│   ├── System.map-4.4.0-62-generic
│   ├── System.map-4.4.0-63-generic
│   ├── System.map-4.4.0-64-generic
│   ├── System.map-4.4.0-66-generic
│   ├── System.map-4.4.0-78-generic
│   ├── vmlinuz-4.4.0-53-generic
│   ├── vmlinuz-4.4.0-59-generic
│   ├── vmlinuz-4.4.0-62-generic
│   ├── vmlinuz-4.4.0-63-generic
│   ├── vmlinuz-4.4.0-64-generic
│   ├── vmlinuz-4.4.0-66-generic
│   └── vmlinuz-4.4.0-78-generic
├── cdrom
├── dev
│   ├── autofs
│   ├── block
│   │   ├── 11:0 -> ../sr0
│   │   ├── 7:0 -> ../loop0
│   │   ├── 7:1 -> ../loop1
│   │   ├── 7:2 -> ../loop2
│   │   ├── 7:3 -> ../loop3
│   │   ├── 7:4 -> ../loop4
│   │   ├── 7:5 -> ../loop5
│   │   ├── 7:6 -> ../loop6
│   │   ├── 7:7 -> ../loop7
│   │   ├── 8:0 -> ../sda
│   │   ├── 8:1 -> ../sda1
│   │   ├── 8:2 -> ../sda2
│   │   └── 8:3 -> ../sda3
│   ├── bsg
│   │   ├── 0:0:0:0
│   │   └── 1:0:0:0
│   ├── btrfs-control
│   ├── bus
│   │   └── usb
│   ├── cdrom -> sr0
│   ├── cdrw -> sr0
│   ├── char

```

ভালো করে লক্ষ্য করলে বুঝতে পারবেন আমার root(/) ডিরেক্টরিতে bin, boot, cdrom, dev এই সমস্ত ডিরেক্টরি বর্তমান যাদের মধ্যে bin ডিরেক্টরিতে শুধুমাত্র বিভিন্ন ফাইল বর্তমান, boot ডিরেক্টরিতে অনেকগুলি ফাইল এবং grub নামের অন্য একটি ডিরেক্টরি বর্তমান, যার মধ্যে আবার অনেক গুলি ফাইল বর্তমান, cdrom ডিরেক্টরিটি খালি, dev নামের ডিরেক্টরির মধ্যে autofs, block, bsg, bus  ইত্যাদি অনেক ডিরেক্টরি বর্তমান।

এভাবেই লিনাক্স বেসড সিস্টেমে ফাইল সাজানো থাকে। বিশদে জানতে ফলো করুন - [Filesystem Hierarchy](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard)

হোম ডিরেক্টরি সাধারণত হয় - /home/<user>. অর্থাৎ এই ডিরেক্টরির মধ্যে আপনার সম্পূর্ণ নিয়ন্ত্রণ, আপনি এখানে ফাইল তৈরী করা, বদলানো বা ডিলিট করা ইত্যাদি সব কাজ করতে পারবেন এবং এটি সাধারণ উপদেশ যে লিনাক্স শেখার সময়ে আপনি আপনার হোম ডিরেক্টরিতে সব কাজ করুন, যাতে করে অন্য কারোর বা অপারেটিং সিস্টেমের  ক্ষতি না হয়।

আসুন এক নজরে দেখে নি বর্তমান ইউজারের হোম ডিরেক্টরিতে কি আছে -

```
totan@Home-Computer ~ $ tree
.
├── abc
├── a.txt
├── data
│   └── test
│       └── linux
├── Desktop
│   ├── google-chrome.desktop
│   ├── Linux Tutorial
│   │   ├── Advanced
│   │   │   ├── debian-handbook.pdf
│   │   │   ├── linux_unleashed__third_edition.pdf
│   │   │   └── slever_et_al_2009.pdf
│   │   └── Basics
│   │       ├── book.pdf
│   │       ├── Emailing UNIXProgrammingEnvironment.pdf
│   │       ├── Introduction_to_Linux-20100512.pdf
│   │       ├── OReilly.Learning.the.vi.and.Vim.Editors.7th.Edition.Jul.2008.pdf
│   │       ├── running_linux_5th_edition.pdf
│   │       ├── TLCL-16.07.pdf
│   │       ├── v7vol1.pdf
│   │       ├── v7vol2a.pdf
│   │       └── v7vol2b.pdf
│   └── xdm.desktop
├── Documents
├── Downloads
│   ├── IE11.Win7.For.Windows.VirtualBox.zip
│   └── IE11 - Win7.ova
├── Music
├── notes.txt
├── Pictures
│   ├── vlcsnap-2017-05-31-20h59m20s614.png
│   ├── vlcsnap-2017-05-31-20h59m20s905.png
│   └── vlcsnap-2017-05-31-20h59m21s229.png
├── Public
├── source
│   └── a
├── Templates
├── Videos
├── VirtualBox VMs
│   ├── FreeBSD
│   │   ├── FreeBSD.vbox
│   │   ├── FreeBSD.vbox-prev
│   │   └── Logs
│   │       ├── VBox.log
│   │       ├── VBox.log.1
│   │       ├── VBox.log.2
│   │       └── VBox.log.3
│   ├── IE11 - Win7
│   │   ├── IE11 - Win7.vbox
│   │   ├── IE11 - Win7.vbox-prev
│   │   └── Logs
│   │       ├── VBox.log
│   │       ├── VBox.log.1
│   │       └── VBox.log.2
│   ├── Oracle Linux
│   │   ├── Logs
│   │   │   ├── VBox.log
│   │   │   ├── VBox.log.1
│   │   │   ├── VBox.log.2
│   │   │   └── VBox.log.3
│   │   ├── Oracle Linux.vbox
│   │   └── Oracle Linux.vbox-prev
│   └── Solaris10_1-13
│       ├── Logs
│       │   ├── VBox.log
│       │   └── VBox.log.1
│       ├── Solaris10_1-13-1.12-linux.vbox
│       ├── Solaris10_1-13.vbox
│       └── Solaris10_1-13.vbox-prev
└── vnc

24 directories, 46 files
totan@Home-Computer ~ $

```

**শেল - শেল হচ্ছে এক বিশেষ ধরণের প্রসেস যেটা অপারেটিং সিস্টেমের সাথে ইউজারের যোগাযোগ বজায় রাখতে সেতুবন্ধনীর মতো কাজ করে। শেল হচ্ছে কম্যান্ড লাইন ইন্টারফেস(CLI), অর্থাৎ আপনার সমস্ত ধরণের ইনপুট আপনি প্রদান করবেন কীবোর্ড এর মাধ্যমে এবং অপারেটিং সিস্টেম আউটপুট হিসেবে টেক্সট ডিসপ্লে করবে। বিশদে জানতে ফলো করুন - [CLI](https://en.wikipedia.org/wiki/Command-line_interface)
সাধারণত বিভিন্ন ধরণের শেল বর্তমান। এক এক ধরণের শেল এক এক ধরণের কাজ করে থাকে বা প্রত্যেক শেল সাধারণ সমস্ত কাজকর্ম করে এবং নিজস্ব কিছু বৈশিষ্ট প্রদান করে। সাধারণ ভাবে যে শেল গুলি ব্যবহার হয় -

bash

C shell

K shell

tcsh

Zsh

সাধারণত: প্রথমে মানুষ জানে না কোনটি ব্যবহার করতে হয়, তাই অপারেটিং সিস্টেম যেটি প্রদান করে, সেটিকেই ব্যবহার করা শুরু করে এবং প্রয়োজন মতো পরে বদলে নেয়। প্রত্যেকের নিজস্ব পছন্দ অনুযায়ী মানুষ এক একটি শেল কে  বেছে নেয়। তবে সাধারণ সমস্ত কাজকর্ম সমস্ত শেলই একই ভাবে করে থাকে। সুতরাং, চিন্তার কোনো বিষয় নেই, আপনার অপারেটিং সিস্টেম প্রদত্ত শেলই এখানকার জন্য যথেষ্ট এবং সাধারণ ভাবে, যেকোনো লিনাক্স বেসড  অপারেটিং সিস্টেমে Ctrl - Alt - t প্রেস করলে ডিফল্ট শেল ওপেন হয়।

এই পোর্স্টটি খুব সাধারণ অর্থে, লিনাক্সের বেসিক থিওরি, এর পরে আমরা শিখে নেবো শেল নিয়ে কাজ করা।

English Version: [https://web.archive.org/web/20190917/http://tech.palashkantikundu.in/2017/07/the-basics-of-unix.html](https://web.archive.org/web/20190917/http://tech.palashkantikundu.in/2017/07/the-basics-of-unix.html)
