---
title: "ফাইল এবং ডিরেক্টরি তৈরী করা"
date: "2017-05-25T23:26:00.000-07:00"
original_url: "http://linux.palash90.in/2017/05/blog-post_70.html"
blog: "লিনাক্স বাংলা গাইড (linux.palash90.in)"
recovered_from: "Atom feed, web.archive.org snapshot 2017-12-03"
---

এই পর্যন্ত আমরা ফাইল সিস্টেম সম্পর্কে বিভিন্ন ধরণের তথ্য আহরণ করেছি। অনেক সময় হয়তো শুধু পড়ে বুঝতে অসুবিধা হয়েছে কোন কম্যান্ড কি কাজে লাগে। তাই এখন আমাদের উদ্দেশ্য এই পর্যন্ত যতগুলো কম্যান্ড আমরা শিখেছি, সেগুলির প্রত্যেকটি টার্মিনালে রান করে বুঝে নেওয়া সেটি কি কাজ করে। হাতেনাতে বিভিন্ন খুঁটিনাটি বিষয় গুলি সম্পর্কে কাজ করলে ফাইল সিস্টেম বুঝতে এবং পরবর্তী সময়ে ফাইল সিস্টেম নিয়ে কাজ করতে সুবিধা হয়।
এইবার আমরা সবার আগে দেখে নেবো কিভাবে নতুন ফাইল এবং ডিরেক্টরি তৈরী করা যায়, তাহলে প্রয়োজনমতো আমরা বিভিন্ন ফাইল এবং ডিরেক্টরি তৈরী করে আমাদের পর্যবেক্ষণ করার কাজে লাগাতে পারবো।
আমি আমার হোম ডিরেক্টরিতে নিজের সুবিধামতো কিছু ডিরেক্টরি তৈরী করে নেবো, আপনিও আপনার হোম ডিরেক্টরিতে এই ডিরেক্টরি গুলি তৈরী করে নিতে পারেন -

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

দেখুন আমি একটি কম্যান্ড রান করেছি যেটিকে **mkdir বলা হয়, বলা বাহুল্য এটা make directory এর শর্টকাট। এই কমান্ডটি রান করে আমরা ডিরেক্টরি তৈরী করতে পারি। এই কমান্ডটির পরে যেকোন নাম লিখে আমরা সেই নামের ডিরেক্টরি তৈরী করতে পারি। যেমন আমি **dir_creation নামের একটি ডিরেক্টরি তৈরী করেছি। **mkdir কমান্ডটি রান করার আগে এবং পরে **ls -l কমান্ডটি রান করে আমরা দেখে নিয়েছি যে ডিরেক্টরিটি সত্যি সত্যি তৈরী হয়েছে কিনা। আপনিও আপনার হোম ডিরেক্টরিতে যেকোনো নামের ডিরেক্টরি তৈরী করতে পারেন এই কমান্ডটি দিয়ে, তবে আপনাকে জানতে হবে ডিরেক্টরি বা ফাইলের নামকরণের কিছু নিয়ম -

ফাইল বা ডিরেক্টরির নামের জন্যে আপনি যেকোনো অক্ষর কীবোর্ড থেকে টাইপ করতে পারেন, শুধুমাত্র forward slash (**/) বা ফরওয়ার্ড স্ল্যাশ বাদ দিয়ে। আমরা আগেও অ্যাবসলিউট পাথ এবং রিলেটিভ পাথ সম্পর্কে জানতে গিয়ে দেখেছি ফরওয়ার্ড স্ল্যাশ দুটি ডিরেক্টরির নাম আলাদা করার জন্যে ব্যবহার করা হয়, তাই এটি আমরা ফাইল বা ডিরেক্টরি নামকরণ করার জন্যে ব্যবহার করতে পারবো না।
তাছাড়াও লিনাক্স সিস্টেমে রুট ডিরেক্টরি বোঝানোর জন্যে আমরা **/ (ফরওয়ার্ড স্ল্যাশ) ব্যবহার করি, তাই আমরা নতুন কোনো ডিরেক্টরি তৈরী করতে পারি না যার নাম **/

কিছু কিছু বিশেষ অক্ষর(!, #, $, &, >, <, space) শেল কমান্ডের সাথে জড়িত, তাই এগুলিকে সরাসরি টাইপ করা যায় না। এই অক্ষর গুলি টাইপ করার জন্যে আমাদের ব্যবহার করতে হয় escape character (এস্কেপ ক্যারেক্টার)। আমরা এর পরে উদাহরণের সাহায্যে escape character সম্পর্কে জেনে নেবো।

যেকোনো আধুনিক লিনাক্স সিস্টেমে ফাইল বা ডিরেক্টরির নাম শুধুমাত্র ১ টি অক্ষর থেকে শুরু করে ২৫৫ টি অক্ষরের সংমিশ্রণ হতে পারে।

একটি ডিরেক্টরির মধ্যে আমরা কখনও দুটো একই নামের ডিরেক্টরি বা ফাইল তৈরী করতে পারবো না। দুটি ভিন্ন ভিন্ন ডিরেক্টরিতে একই নামের ফাইল থাকতে পারে, অসুবিধা নেই কিন্তু একই ডিরেক্টরির ভেতরে দুটি ফাইলের নাম এক হতে পারে না। অর্থাৎ যেকোনো দুটি ফাইলের অ্যাবসলিউট পাথ কখনো একই হতে পারে না। এই নিয়মটির সাহায্যে ভেবে দেখার চেষ্টা করুন, প্রথম নিয়মটি কিভাবে যথার্থ। আমরা **/ নামের কোনো ফাইল বা ডিরেক্টরি তৈরী করতে পারি না কারণ সেটা ইতিমধ্যেই ফাইল সিস্টেমে বর্তমান।

এই নিয়ম অনুযায়ী আমরা বিভিন্ন নামের ফাইল তৈরী করতে পারি। নিয়মে সমস্ত ক্যারেক্টার ব্যবহার করার অনুমতি থাকলেও সাধারণত দৈনন্দিন ব্যবহারের জন্যে ইংরাজীর ছোট হরফের অক্ষর, সংখ্যা এবং underscore (**_) এর সমন্বয়ই সবথেকে বেশি চোখে পড়ে ফাইল এবং ডিরেক্টরির নামের ক্ষেত্রে। আধুনিক লিনাক্স সিস্টেমে ফাইল বা ডিরেক্টরির নামে যেকোনো ভাষার অক্ষর ব্যবহার করার সুবিধাও রয়েছে। আর একটা কথা মাথায় রাখতে হবে ফাইল বা ডিরেক্টরি নামের ক্ষেত্রে, space যত সম্ভব কম ব্যবহার করাই ভালো, কারণ অনেক সময় এটি ভ্রান্তির কারণ হয়ে ওঠে এবং সেই নামের ফাইল বা ডিরেক্টরি নিয়ে কাজ করার জন্যে আমাদের escape character ব্যবহার করতে হয়, যা সময়ে সময়ে বেদনাদায়ক হয়ে ওঠে।
আসুন এবার তাহলে আমরা **dir_creation ডিরেক্টরির মধ্যে কিছু নতুন ডিরেক্টরি তৈরী করে দেখে নি সমস্ত নিয়ম গুলিকে -
![image](https://1.bp.blogspot.com/-GVqhiOVvoyQ/WfyEqU9KPDI/AAAAAAAAJC4/wE1RrzHwOKwNLvt4QcFLnrUhFvqPcpzlwCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-11-03%2B20-29-34.png)

![image](https://1.bp.blogspot.com/-4G_bTZeQRj4/WfyEqT7G24I/AAAAAAAAJC0/UJpyVlYGuY0-UFGyT9_9UN0QqGxIA5aNgCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-11-03%2B20-29-59.png)

![image](https://1.bp.blogspot.com/-KMAi70m28OQ/WfyEqRd3iSI/AAAAAAAAJCw/w7iETP3yO0sRQcdYqX6X3mwnjo-vLTp5wCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-11-03%2B20-30-14.png)

আমি এখানে ৩ টি ভিন্ন ভিন্ন লিনাক্স/উনিক্স সিস্টেমে **/ নামের ডিরেক্টরি তৈরী করতে চেয়েছি কিন্তু আমাকে কোনো সিস্টেম সমর্থন করে নি এবং প্রত্যেক ক্ষেত্রেই আমাকে জানানো হয়েছে যে **/ ডিরেক্টরি ইতিমধ্যেই বর্তমান।

এবার আমরা চেষ্টা করবো বিভিন্ন ডিরেক্টরি তৈরী করতে যেগুলির নামে বিশেষ অক্ষরের প্রয়োগ আছে -

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

উপরের আউটপুট দেখে নিশ্চয় আপনার ধারণা হয়ে গেছে যে এই সমস্ত বিশেষ অক্ষর গুলি বিশেষ কোনো  কাজ সম্পন্ন করছে তাই ফাইল বা ডিরেক্টরি নামে এগুলি ব্যবহার করলে সেটি ঠিকঠাক কাজ করছে না। যেমন আপনি হয়তো ধারণা করে ফেলেছেন যে mkdir ~ কাজ  করছে না, তার কারণ লিনাক্সে **~ বলতে হোম ডিরেক্টরি বোঝায়। তাহলে এখন প্রশ্ন হলো, লিনাক্সে ফাইল নামের ক্ষেত্রে যদি সমস্ত অক্ষরকেই ব্যবহার করা যায় তো আমরা সেই নামের ফাইল বা ডিরেক্টরি তৈরী কি করে করবো যেখানে বিশেষ অক্ষরের প্রয়োগ আছে যেমন **~, !d , !kj, #, #kj  ইত্যাদি ইত্যাদি। যদিও এই ধরণের নামের প্রচলন খুবই কম বা প্রায় নাই বললেই চলে তবুও আমরা দেখে নেবো কিভাবে এই বিশেষ অক্ষরগুলির সমন্বয়ে তৈরী নামের ফাইল বা ডিরেক্টরি তৈরী করতে পারি-

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

দেখুন আমরা যখনই কোনো একটি বিশেষ অক্ষর ব্যবহার করতে চাইছি নামের মধ্যে আমরা কিন্তু সেই অক্ষরটির আগে backward slash (**\) ব্যবহার করছি। এই backward slash (**\) কে বলা হয় escape character, অর্থাৎ আমরা escape character ব্যবহার করে বিশেষ অক্ষরগুলির সমন্বয়ে তৈরী নাম ব্যবহার করতে পারি। আরো লক্ষ্য করুন, আমি একটি ডিরেক্টরি তৈরী করেছি যার নাম বাংলা বর্ণমালা থেকে নেওয়া হয়েছে।

এবার আমি চেষ্টা করবো একটি ফাইল তৈরী করার যেটার ফাইল নামের অক্ষর সংখ্যা ২৫৬, আপনি তার বেশি সংখ্যার নাম দিয়েও ফাইল তৈরী করার চেষ্টা করতে পারেন -

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

দেখুন, প্রথমবার অক্ষর সংখ্যা ছিল ২৫৬, তাই লিনাক্স আমাদের দেখালো touch: cannot touch 'l6ecYSSPVBgKxMPKQ0k4D1 ... কিন্তু যখনই আমি শেষ অক্ষরটা সরিয়ে দিলাম, লিনাক্স কিন্তু ফাইলটি তৈরী করে দিলো ডিরেক্টরিতে। এক্ষেত্রে আপনাদের জানা দরকার, **touch হলো একটি কম্যান্ড যেটি একটি ফাঁকা ফাইল আপনার জন্যে তৈরী করে দেবে। **touch টাইপ করার পর একটি স্পেস দিয়ে যেকোনো ফাইল নাম টাইপ করে এন্টার করলেই এটি একটি ফাঁকা ফাইল তৈরী করে দেবে, আর যদি ফাইলটি ইতিমধ্যেই বর্তমান থাকে তাহলে ফাইলটির টাইম স্ট্যাম্প পরিবর্তন করে দেবে (খুব শীঘ্রই আমরা জেনে নেবো কি হয় এই টাইম স্ট্যাম্প)।
এইবারে আমরা দেখে নেবো সর্বশেষ নিয়মের প্রয়োগ -

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

প্রথমে আমি একটি ফাইল তৈরী করলাম, যার নাম **a এবং আমি চেষ্টা করলাম সেই একই নামের একটি ডিরেক্টরি তৈরী করতে কিন্তু আমি সফল হলাম না, সিস্টেম আমাকে এরর দেখালো - mkdir: cannot create directory ‘a’: File exists।কিন্তু আমি তো ডিরেক্টরি তৈরী করতে চেয়েছিলাম, তাহলে ফাইল দেখাচ্ছে এটা একটা বেশ অদ্ভুত ব্যাপার তাই না। আসলে লিনাক্স সিস্টেমে সব কিছুই ফাইল হিসেবে বিবেচিত হয়। ডিরেক্টরি একটি বিশেষ ধরণের ফাইল বই আর কিছু না; একটি বিশেষ ফাইল যেটি অন্য কিছু ফাইল বা ডিরেক্টরির সমন্বয়।
ফিরে আসি আমাদের আগের আলোচনায়, একই ঘটনা ঘটলো যখন আমি চেষ্টা করলাম 4 এবং 6  নাম দিয়ে ডিরেক্টরি তৈরী করতে। কিন্তু যখন আমরা প্রথমে 8 নামের একটি ডিরেক্টরি তৈরী করলাম এবং পরে **touch কম্যান্ড ব্যবহার করলাম সিস্টেম কোনো এরর দেখালো না।

হয়তো ভাবছেন এই রকম কেন হলো, একই নামের একটি ফাইল এবং একটি ডিরেক্টরি কিভাবে তৈরী হয়ে গেলো একই  ডিরেক্টরির মধ্যে। আপনি ভুল ভাবছেন, মোটেও নতুন ফাইল তৈরী হয় নি 8 নামের। ভালো করে **ls কমান্ডের অউটপুটগুলি লক্ষ্য করুন, তাহলেই বুঝতে পারবেন ব্যাপারটা ঠিক কি ঘটেছে।

এই একটা প্রশ্নের উত্তর খুঁজতে এখনো অবধি এখানে ফাইল সিস্টেম সম্পর্কিত যত জিনিস আলোচনা করা হয়েছে, সেই সমস্ত কিছুর সাহায্য প্রয়োজন। পার্থক্যটা খুঁজে বের করাটা আপনার হোমওয়ার্ক।
