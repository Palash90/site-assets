# ফাইল সিস্টেমের খুঁটিনাটি


আমরা দেখে নিয়েছি কিভাবে একটি ডিরেক্টরি থেকে অন্য ডিরেক্টরির ফাইল দেখতে পাবো, দেখে নিয়েছি কিভাবে ফাইল সিস্টেমের বিভিন্ন জায়গায় আমরা নেভিগেট করতে পারি, কিভাবে সেই ডিরেক্টরির বিভিন্ন ফাইল দেখতে পারি, আমাদের  অ্যাবসলিউট পাথ এবং রিলেটিভ পাথ সম্পর্কে ধারণা এখন স্বচ্ছ। এখন আমরা আরো একটু গভীরে প্রবেশ করার চেষ্টা করবো। লিনাক্স সিস্টেমে আমরা বিভিন্ন ধরণের ফাইল সেভ করে রাখতে পারি। আমরা যে কোনো ধরণের ফাইল সেভ করে রাখলেও, লিনাক্স সিস্টেম জানে না কোন ফাইল কি কাজ করে বা কোন ফাইলের মধ্যে কি তথ্য সেভ করা আছে। কোন ফাইলের মধ্যে কি তথ্য সেভ করা আছে এবং সেই তথ্য কিভাবে ইউজারের সামনে উপস্থাপন করতে হবে সেটা যে প্রোগ্রাম দিয়ে ফাইলটিকে ওপেন করা হয়, সেই প্রোগ্রামের দায়িত্ব বুঝে নেবার। উদাহরণস্বরূপ, আমরা কোনো একটি ডিরেক্টরির মধ্যে ছবি, ভিডিও, প্রোগ্রামিং ফাইল এবং আরো অন্য ধরণের ফাইল সেভ করে রাখতে পারি, কিন্তু আমরা যখন যে প্রোগ্রাম দিয়ে সেই ফাইলটিকে ওপেন করবো, সেই প্রোগ্রামটি সেই ফাইলের তথ্য আমাদের সামনে উপস্থাপন করবে।

লিনাক্স সিস্টেম যদিও জানে না তার কাছে কি ধরণের ফাইল বর্তমান, তাও আমরা একটি প্রোগ্রামের সাহায্যে যেকোনো ফাইল সম্পর্কে একটি ধারণায় উপনীত হতে পারি যে ফাইলটি কি এবং সেটি কি করতে পারে। এই বিশেষ প্রোগ্রামটির নাম **file. আমরা কিছু উদাহরণের সাথে এই কমান্ডটির সম্পর্কে একটু ধারণা করে নিতে পারি -

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

আমরা কোনো একটি ডিরেক্টরির মধ্যে নেভিগেট করে সেই ডিরেক্টরির একটি ফাইল সম্পর্কে ধারণা করে নিতে পারি **file কমান্ডটির সাহায্যে, অনেক সময় এটি বেশ কাজে  আসে,আমরা কোনো একটি ফাইল সম্পর্কে ধারণা করে প্রয়োজনীয় কমান্ডের সাহায্যে সেই ফাইলটিকে ওপেন করতে পারি। ফাইল কমান্ডটিকে আমরা এবার আরো কিছু উদাহরণের সাথে দেখে নেবো -

```
totan@Home-Computer ~ $ file /mnt/sda3/Media/E-books/Matlab\ Techniques.pdf
/mnt/sda3/Media/E-books/Matlab Techniques.pdf: PDF document, version 1.3
totan@Home-Computer ~ $ file notes/ch1.3.txt
notes/ch1.3.txt: ASCII text, with very long lines
totan@Home-Computer ~ $

```

এখানে আমরা ফাইল কমান্ডটিকে অ্যাবসলিউট পাথ এবং রিলেটিভ পাথের সাহায্যে ব্যবহার করলাম। সেইভাবেই আমরা আরো কিছু উদাহরণ দিয়ে দেখে নেবো ফাইল কমান্ডটির বিভিন্ন আউটপুট -

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

এইভাবে ফাইল কম্যান্ড আমাদের অনেক সময় ফাইল কোন ধরণের সেটা বুঝতে সাহায্য করে এবং অনেক সময় সেই ফাইলের অন্যান্য আরো বৈশিষ্ট আমাদের সামনে তুলে ধরে।

এইবার আমরা ফাইল সিস্টেমের বিষয়ে কিছু প্রয়োজনীয়  তথ্য জেনে নেবো এবং এগুলি আমাদের সব সময় মাথায় রাখা উচিত -

#### ফাইল বা ডিরেক্টরির নাম কেস সেনসিটিভ

অর্থাৎ লিনাক্স সিস্টেমে file1.txt, File1.txt, fIle1.txt, fiLe1.txt, filE1.txt ভিন্ন ভিন্ন ফাইল। তাই আমরা যদি কোনো ফাইল file1.txt নামে সেভ করি এবং  File1.txt টাইপ করে ওপেন করতে চাই, আমরা সেটি করতে পারবো না। আসুন একটা উদাহরণ দিয়ে দেখে নি -

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

লক্ষ করুন আমার হোম ডিরেক্টরিতে file1.txr নামের একটি ফাইল আছে কিন্তু আমি যখন ***less File1.txr টাইপ করে ফাইলটি ওপেন করতে  গেলাম,লিনাক্স আমাকে একটি এরর শো করলো এবং আমাদের বুঝিয়ে দিলো যে, আমি যে ফাইলটি ওপেন করতে চাইছি সেটা ওই ডিরেক্টরিতে নেই।

#### লিনাক্স সিস্টেমে ফাইলের এক্সটেনশন না থাকলেও চলে

একটি ফাইলের এক্সটেনশন বলতে যেকোনো ফাইল নামের শেষ **. এর থেকে বাকি অংশটিকে বোঝায়। নিচের কিছু ফাইল নাম থেকে এক্সটেনশন সম্পর্কে একটা ধারণা তৈরী করার চেষ্টা করবো -

```
dr-xr-xr-x  2 palash palash     4096 Aug 15  2015 Stellar Astrophysics
dr-xr-xr-x  2 palash palash     4096 Aug 15  2015 The Hackers Underground Handbook
-r-xr-xr-x  1 palash palash   202453 May  2  2011 The Pythagorean Theorem.pdf
-r-xr-xr-x  1 palash palash 10477660 Mar 15  2011 ebooksclub.org__Modern_Graph_Theory.djvu

```

এখানে  Stellar Astrophysics,  The Hackers Underground Handbook ডিরেক্টরি, বাকিগুলি ফাইল এবং The Pythagorean Theorem.pdf ফাইলটির এক্সটেনশন .pdf এবং ebooksclub.org__Modern_Graph_Theory.djvu ফাইলটির এক্সটেনশন .djvu।
অনেক ক্ষেত্রে এক্সটেনশন ফাইল চেনার জন্যে এবং প্রয়োজনীয় কম্যান্ড রান করে ফাইলটিকে ওপেন করতে কাজে লাগে কিন্তু লিনাক্স সিস্টেমে ফাইলের এক্সটেনশন থাকা জরুরি নয়, একটা ফাইলের কোনো এক্সটেনশন নাও থাকতে পারে। উদাহরণস্বরূপ নিচের ফাইল গুলির কোনো এক্সটেনশন নেই -

```
-rw-r--r--  1 totan totan        2 Oct 27 20:59 ch1
-rw-r--r--  1 totan totan        4 Oct 27 21:00 ch2
-rw-r--r--  1 totan totan        4 Oct 27 21:00 ch3
-r-xr-xr-x  1 totan totan 29436482 Oct 27 21:00 ch4

```

কিন্তু মজার জিনিস এটাই যে ফাইল এক্সটেনশন না থাকা সত্ত্বেও আমরা ফাইল কম্যান্ড রান করে ফাইলটি কি ধরণের সেটা জেনে নিতে পারি-

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

#### হিডেন ফাইল

যদি কোনো ফাইল বা ডিরেক্টরির নামের প্রথম অক্ষরটি **. হয় তাহলে লিনাক্স সিস্টেমে সেই ফাইলটিকে হিডেন ফাইল বলা হয় এবং সাধারণ ভাবে এই ফাইলগুলি দেখা যায় না। হিডেন ফাইল দেখতে অন্য ব্যবস্থা অবলম্বন করতে হয় (পরে আমরা দেখে নেবো কিভাবে সেগুলি করতে হয়)। সাধারণত এটা সিকিউরিটির জন্যে করা হয়। সাধারণত যেকোনো লিনাক্স সিস্টেমের অনেক ধরণের কনফিগারেশন থাকে, যেগুলির ভিত্তিতে সিস্টেমটি চলে, তাই কোনো সাধারণ ইউজার এসে এই ফাইলগুলি পাল্টে দিলে বা ডিলিট করে দিলে সিস্টেমটি ঠিক ভাবে চলবে না, তাই এই ধরণের ফাইলগুলি সাধারণত হিডেন ফাইল হয়।  যদিও ফাইলের সিকিউরিটির জন্যে আরো অন্য রাস্তা নেওয়া যায়, হিডেন ফাইলও ক্ষেত্র বিশেষে বেশ উপযোগী। 
তাছাড়া আরো একটা কারণ থাকে এর পেছনে, সমস্ত ফাইল সব সময় প্রয়োজনীয় হয় না, অপ্রয়োজনীয় ভাবে সব সময় সমস্ত ফাইল সামনে এলে অনেক সময় প্রয়োজনীয় ফাইল খুঁজে পেতে অসুবিধা হয়।
উদাহরণের সাহায্যে এবার দেখে নি কিভাবে হিডেন ফাইল কাজের সুবিধা করে -

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

এটি হচ্ছে সাধারণ ভিউ, এবার আমি হিডেন ফাইলও দেখতে চাই, তাহলে আমি হিডেন ফাইল দেখার কম্যান্ড রান করে এখন দেখে নেবো আমার হোম ডিরেক্টরিতে বস্তুত কি কি ফাইল বর্তমান -

```
drwxr-xr-x 47 totan totan     4096 Oct 27 21:03 .
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

দুটি আউটপুটের মধ্যে পার্থক্য দেখলেই বুঝতে পারবেন কেন দৈনন্দিন ব্যবহারের জন্যে হিডেন ফাইল উপযোগী।

এর পরে আমরা দেখে নেবো ফাইল সিস্টেমের বিভিন্ন কম্যান্ড যেগুলি দিয়ে আমরা আরো অনেক কিছু করতে পারি।
