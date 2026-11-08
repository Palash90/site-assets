---
title: "bc - ফাইল এবং প্রোগ্রামিং"
date: "2016-07-01T23:57:00.000-07:00"
original_url: "http://linux.palash90.in/2016/06/bc.html"
blog: "লিনাক্স বাংলা গাইড (linux.palash90.in)"
recovered_from: "Atom feed, web.archive.org snapshot 2017-12-03"
---

আমরা দেখে নিয়েছি কিভাবে আমরা bc এর সাহায্য নিয়ে বিভিন্ন ধরণের হিসেব করতে পারি, কিভাবে আমরা function এর সাহায্য নিয়ে একই সূত্র বিভিন্ন মানের উপর প্রয়োগ করতে পারি। কিন্তু আমরা দেখেছিলাম, আমাদের function গুলি bc থেকে exit করার পর আর পরের বার bc রান করলে চালাতে পারি না, তাই আমরা এখন দেখে নেবো কিভাবে আমরা ফাইলে সেভ করে সেগুলি চালাতে পারি। bc একসাথে অনেক গুলি ফাইলও ইনপুট হিসেবে নিতে পারে। এবার আমরা দেখে নেবো কিভাবে ফাইল তৈরী করে, তাতে কিছু function লিখে সেটিকে bc তে ব্যবহার করবো আমরা -

```
totan@Home-Computer ~ $ mkdir bc
totan@Home-Computer ~ $ cd bc
totan@Home-Computer ~/bc $ vi custom_lib.bc

```

আমি প্রথমে আমার home ডিরেক্টরিতে bc নামের আরও একটি ডিরেক্টরি বানিয়েছি এবং তাতে একটি ফাইল বানিয়েছি, যার নাম custom_lib.bc এবং সেই ফাইলে আমি নিচের লাইনগুলি লিখে ফাইলটি সেভ করে রেখেছি ব্যবহারের জন্যে -

```
scale=40;
pi=22/7;
define tan(x) {return s(x)/c(x); }

```

এবার bc রান করার সময়ে আমরা ফাইলটি ইনপুট দেব এবং আমাদের tan function টি রান করবো -

```
otan@Home-Computer ~/bc $ bc custom_lib.bc
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.
tan(90)
Runtime error (func=tan, adr=4): Function s not defined.

```

কি হলো ?
error কেন?
আসলে আমরা আমাদের ফাইলটি ইনপুট দিলেও, সেটি যখন রান করছে, তখন তার s এবং c function টি দরকার, যেটি math library তে লেখা রয়েছে কিন্তু আমরা bc -l রান করি নি, শুধুমাত্র bc রান করে তাতে আমাদের ফাইলটি ইনপুট দিয়েছি, তাই এই error দেখাচ্ছে। এইবার আমরা ঠিকঠাক ভাবে রান করে দেখবো কি হয় -

```
totan@Home-Computer ~/bc $ bc -l custom_lib.bc
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.

pi
3.1428571428571428571428571428571428571428

scale
40

tan(pi/2)
-1581.6660411069837079729290816982181591162519

```

আসুন এবার আমরা সমস্ত ত্রিকোণমিতির সূত্র গুলি function আকারে প্রকাশ করে ফাইলে সেভ করি -

```
scale=40;
pi=22/7;

define degree(radian) { return radian * (180/(22/7)); }
define radian(degree) { return degree * ((22/7)/180); }

define sin(x) { return s(x); }
define cos(x) { return c(x); }
define tan(x) { return sin(x)/cos(x); }
define cot(x) { return cos(x)/sin(x); }
define sec(x) { return 1/cos(x); }
define cosec(x) { return 1/sin(x); }

define arctan(x) { return a(x); }
define arcsin(x) { return 2*arctan(x/(1+sqrt(1-x^2))); }
define arccos(x) { return pi/2 -  arcsin(x); }
define arccot(x) { return pi/2 - arctan(x); }
define arccsc(x) { return arcsin(1/x); }
define arcsec(x) { return arccos(1/x); }

```

এইবার আমাদের কাছে একটি সম্পূর্ণ ত্রিকোণমিতিক লাইব্রেরি আছে এবং আমরা সেটিকে ব্যবহার করবো -

```
totan@Home-Computer ~/bc $ bc -l custom_lib.bc
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.

pi
3.1428571428571428571428571428571428571428

degree(pi)
179.9999999999999999999999999999999999999997

degree(pi/2)
89.9999999999999999999999999999999999999998

radian(1.57)
.0274126984126984126984126984126984126983

radian(89.99999999999999999999999999999998)
1.5714285714285714285714285714285710793596

radian(179.99999999999999999999999999999998)
3.1428571428571428571428571428571425079256

tan(pi)
.0012644899412946341567363163883638298730

cot(pi)
790.8327044311328950385320768793316826822764

sin(pi/2)
.9999998001333682524818408858972908885417

cos(pi/2)
-.0006322445915532736545183507752145567302

cos(pi/4)
.7068832136245869031428194525782683945745

sin(pi/4)
.7073302780849810594563841044291318942708

sec(pi/4)
1.4146608389134590576641966960370960477362

cosec(pi/4)
1.4137667098139641735735833689442185007077

cos(radian(30))
.8659200104474300306216745503726937450452

cos(pi/6)
.8659200104474300306216745503726937450443

arccos(0.8659200104)
.5244417685380240683932317158122716760510

degree(arccos(0.8659200104))
30.0362103799050148261578164510664687192850

```

এইভাবে আমরা ফাইলের সাহায্য নিয়ে নিজেদের লাইব্রেরি তৈরী করে ব্যবহার করতে পারি।
কিন্তু সবকিছুর পরেও আপনার মনে হচ্ছে, এই সব করেই বা লাভ কি যদি সব মান আমরা ঠিকঠাক ভাবে হিসেব করতেই না পারি।  তাহলে আপনাকে বলি, সমস্ত ইনপুটের জন্যে না পারলেও কিছু কিছু বিশেষ ইনপুটের জন্যে অবশ্য আপনি প্রয়োজনীয় আউটপুট return করতে পারেন, যেমন cos(pi/2), sin(pi/2)  ইত্যাদি ইনপুটের জন্যে আপনি সঠিক মান return করতে পারেন। তার জন্যে আপনাকে আরও একটু প্রোগ্রামিং শিখতে হবে। আসুন দেখে নি আমরা কিভাবে বিশেষ কিছু ইনপুটের জন্যে আলাদা return করতে পারি।
আমি আগের ফাইলটি পরিবর্তন করেছি -

```
scale=40;
pi=22/7;

define degree(radian) { return radian * (180/(22/7)); }
define radian(degree) { return degree * ((22/7)/180); }

define sin(x) {
if (x == pi/2){
return 1;
} else {
return s(x);
}
}
define cos(x) {
if (x == pi/2){
return 0;
} else {
return c(x);
}
}
define tan(x) {
if (x == pi/2){
return 10^40;
} else {
return sin(x)/cos(x);
}
}
define cot(x) {
if (x == 0){
return 10^40;
} else {
return cos(x)/sin(x);
}
}
define sec(x) {
if (x == pi/2){
return 10^40;
} else {
return 1/cos(x);
}
}
define cosec(x) {
if (x == 0){
return 10^40;
} else {
return 1/sin(x);
}
}

define arctan(x) { return a(x); }
define arcsin(x) { return 2*arctan(x/(1+sqrt(1-x^2))); }
define arccos(x) { return pi/2 -  arcsin(x); }
define arccot(x) { return pi/2 - arctan(x); }
define arccsc(x) { return arcsin(1/x); }
define arcsec(x) { return arccos(1/x); }

```

এখন দেখুন, প্রত্যেকটি বিশেষ মানের ইনপুটের জন্যে আমি  if ব্যবহার করেছি, আসলে এটি একটি keyword, যেটি bc কে বোঝাতে সাহায্য করে যে আমরা একটি বিশেষ পরিস্থিতির জন্যে চেক করছি এবং সেই বিশেষ পরিস্থিতি এলে আমরা অন্য ভাবে প্রোগ্রাম রান করবো নাহলে অন্য আরেক ভাবে প্রোগ্রাম রান করবো, এই বিশেষ পরিস্থিতিকে বন্ধনীর মধ্যে লিখবো। যেমন tan এর ক্ষেত্রে বিশেষ পরিস্থিতি যখন ইনপুট pi/2, তখন আমরা আমরা ইনফিনিটি (বাস্তবিক ক্ষেত্রে ইনফিনিটি সম্ভব নয়, তাই আমরা ১০^৪০ return করছি) return করার চেষ্টা করছি অন্য ক্ষেত্রে sin(x)/cos(x)। একই ভাবে sin, cos, cot, sec, cosec প্রত্যেকের ক্ষেত্রেই নিজস্ব বিশেষ অবস্থা আছে। আমরা এবার bc রান করে কিছু মান গণনা করবো -

```
totan@Home-Computer ~/bc $ bc custom_lib.bc -l
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.

sin(0)
0

sin(pi/2)
1

sin(pi/3)
.8662360750607195043408695349267950975451

sin(pi/4)
.7073302780849810594563841044291318942708

sin(pi/6)
.5001825021996697823102745005453209790920

cos(pi/6)
.8659200104474300306216745503726937450443

cos(0)
1.0000000000000000000000000000000000000000

cos(pi/2)
0

tan(pi/2)
10000000000000000000000000000000000000000

cot(0)
10000000000000000000000000000000000000000

tan(pi/4)
1.0006324445845895900615571597665295784979

cot(pi/4)
.9993679551487537983317012806569424613799

```

কিন্তু তার পরেও আমরা দেখতে পাচ্ছি যে tan(pi/4) বা cot(pi/4)  এর মান যথাযথ নেই, তাই আমরা ফাইলটি আবার পরিবর্তন করবো -

```
scale=40;
pi=22/7;

define degree(radian) { return radian * (180/(22/7)); }
define radian(degree) { return degree * ((22/7)/180); }

define sin(x) {
if (x == pi/2){
return 1;
} else {
return s(x);
}
}
define cos(x) {
if (x == pi/2){
return 0;
} else {
return c(x);
}
}
define tan(x) {
if (x == pi/2){
return 10^40;
} else if (x == pi/4){
return 1;
} else {
return sin(x)/cos(x);
}
}
define cot(x) {
if (x == 0){
return 10^40;
} else if (x == pi/4){
return 1;
} else {
return cos(x)/sin(x);
}
}
define sec(x) {
if (x == pi/2){
return 10^40;
} else {
return 1/cos(x);
}
}
define cosec(x) {
if (x == 0){
return 10^40;
} else {
return 1/sin(x);
}
}

define arctan(x) { return a(x); }
define arcsin(x) { return 2*arctan(x/(1+sqrt(1-x^2))); }
define arccos(x) { return pi/2 -  arcsin(x); }
define arccot(x) { return pi/2 - arctan(x); }
define arccsc(x) { return arcsin(1/x); }
define arcsec(x) { return arccos(1/x); }

```

এই ক্ষেত্রে দেখুন আমরা দুটি ভিন্ন অবস্থা বোঝানোর জন্যে else if ব্যবহার  করেছি। অর্থাৎ if এর সাথে পরিস্থিতি না মিললে else if এর জন্যে চেক হবে, সেটাও নাহলে তখন else এর প্রোগ্রামটি রান করবে। যেমন tan(x) ক্ষেত্রে ইনপুট যদি pi/2 হয় তাহলে ইনফিনিটি রিটার্ন করবে, pi/4 হলে ১ রিটার্ন করবে এবং অন্যান্য অবস্থায় সূত্র অনুযায়ী মান রিটার্ন করবে। আসুন দেখে নি একবার চালিয়ে -

```
totan@Home-Computer ~/bc $ bc custom_lib.bc -l
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.

tan(pi/2)
10000000000000000000000000000000000000000

tan(0)
0

cot(0)
10000000000000000000000000000000000000000

cot(pi/2)
0

cot(pi/4
(standard_in) 11: syntax error

cot(pi/4)
1

tan(pi/4)
1

tan(pi/6)
.5776313010034497200431518875171948073868

tan(pi/10)
.3250595004727384774258910058063649166791

tan(pi/180)
.0174620920098073892786660773694508440827

sin(pi/180)
.0174594303072945382761249742439511767753

sin(pi/pi)
.8414709848078965066525023216302989996225

```

```

```

```
sin(radian(90))
.9999998001333682524818408858972908885418

cos(radian(90))
-.0006322445915532736545183507752145567248

tan(radian(90))
-1581.6660411069837079729290816982181726254171
```

আপনি নিশ্চয় শেষের কয়েকটি লাইন লক্ষ্য করেছেন, হয়তো অনেকে এর কারণটা বুঝতেও পেরেছেন। আসলে degree * ((22/7)/180) একেবারে radian মানের সমান হয় না, তাই tan function টি সঠিক প্রোগ্রাম চালাতে পারছে না। আসুন তাহলে দেখে নি কিভাবে এটাকে ঠিক করা যায়। এখানে আমরা আরো একটি operator এর সাহায্য নেবো, যেটাকে বলা হয় boolean or অপারেটর। যদি আপনি boolean algebra সম্পর্কে না জেনে থাকেন, তাহলে আমি একটি ভিডিও এখানে এমবেড করবো, যেটি আপনাকে বুঝতে সাহায্য করবে -
&lt;/iframe&gt;এইবার আমরা দেখে নেবো কিভাবে আমরা এই boolean or অপারেশন ব্যবহার করবো bc তে। তার জন্যে আমি আবার আগের প্রোগ্রামটিকে একটু পরিবর্তন করবো -

```
scale=40;
pi=22/7;

define degree(radian) { return radian * (180/(22/7)); }
define radian(degree) { return degree * ((22/7)/180); }

define sin(x) {
if (x == pi/2 || x == radian(90)){
return 1;
} else {
return s(x);
}
}
define cos(x) {
if (x == pi/2 || x == radian(90)){
return 0;
} else {
return c(x);
}
}
define tan(x) {
if (x == pi/2 || x == radian(90)){
return 10^40;
} else if (x == pi/4 || x == radian(45)){
return 1;
} else {
return sin(x)/cos(x);
}
}
define cot(x) {
if (x == 0){
return 10^40;
} else if (x == pi/4 || x == radian(45)){
return 1;
} else {
return cos(x)/sin(x);
}
}
define sec(x) {
if (x == pi/2 || x == radian(90)){
return 10^40;
} else {
return 1/cos(x);
}
}
define cosec(x) {
if (x == 0){
return 10^40;
} else {
return 1/sin(x);
}
}

define arctan(x) { return a(x); }
define arcsin(x) { return 2*arctan(x/(1+sqrt(1-x^2))); }
define arccos(x) { return pi/2 -  arcsin(x); }
define arccot(x) { return pi/2 - arctan(x); }
define arccsc(x) { return arcsin(1/x); }
define arcsec(x) { return arccos(1/x); }

```

এইবার এই পরিবর্তিত প্রোগ্রামটিকে ইনপুট দিয়ে আমরা আবার কিছু মান গণনা করবো -

```
totan@Home-Computer ~/bc $ bc -l custom_lib.bc
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.

sin(radian(90))
1

sin(pi/2)
1

cos(radian(90))
0

cos(pi/2)
0

cos(pi/4)
.7068832136245869031428194525782683945745

tan(pi/4)
1

tan(radian(45))
1

cot(radian(45))
1

sec(radian(90))
10000000000000000000000000000000000000000

cosec(0)
10000000000000000000000000000000000000000

cosec(90)
1.1185724071637082998393926874013871636495

cosec(radian(90))
1.0000000000000000000000000000000000000000

```

এইভাবে প্রোগ্রামিং করে এবং বিভিন্ন অপারেটর ব্যবহার করে আপনি বিভিন্ন ধরনের প্রোগ্রামিং করতে পারেন। আমাদের পরের পোস্টে দেখে নেবো কিভাবে আমরা লুপের সাহায্যে ফ্যাক্টোরিয়াল নির্ণয় করতে পারি - [bc - লুপ](http://linux.palash90.in/2016/06/bc_13.html) 
