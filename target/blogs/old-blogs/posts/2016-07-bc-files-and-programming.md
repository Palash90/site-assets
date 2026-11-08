---
title: "bc - Files and Programming"
date: "2016-07-17T12:27:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045107)"
---

We have seen how we can calculate different values of different mathematical operations with the help of **bc and in **bc how we can apply the same formula to different values ​​with the help of function. But we saw that we can not run our functions second time after **bc stops running, so we will now see how we can save them in a file and run them across different **bc sessions. **bc can take multiple files as input at the same time. Now we will take a look at how to create a file, write some function in it and use it in bc.

```
totan @ home-computer ~ $ mkdir bc
totan @ home-computer ~ $ cd bc
totan @ home-computer ~ / bc $ vi custom_lib.bc

```

**I first created another directory named **bc in my home directory and created a file named **custom_lib.bc and in that file I have saved the following lines -

```
scale = 40;
pi = 22/7;
define tan ( x )  {return s ( x ) / c ( x ) ; }

```

**Now when we run **bc we will input the file and run our tan function -

```
totan@Home-Computer ~/bc $ bc custom_lib.bc
bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'.
tan(90)
Runtime error (func=tan, adr=4): Function s not defined.

```

What happened ? 
Why an error ?
We only provided input of our file but when it runs, it needs s and c function, which is written in math library but we did not run **bc -l, we just input our file to **bc, so this error shows. This time we will run properly and see what happens -

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

**Let us now convert all the trigonometric formulas in the form of function and save it to the file -

```
scale = 40;
pi = 22/7;
define degree ( radian )  {  return radian * ( 180 / ( 22/7 )) ; }
define radian ( degree )  {  return degree * (( 22/7 ) / 180 ) ; }

define sin ( x )  {  return s ( x ) ; }
define cos ( x )  {  return c ( x ) ; }
define tan ( x )  {  return sin ( x ) / cos ( x ) ; }
define cot ( x )  {  return cos ( x ) / sin ( x ) ; }
define sec ( x )  {  return 1 / cos ( x ) ; }
define cosec ( x )  {  return 1 / sin ( x ) ; }

define arctan ( x )  {  return a ( x ) ; }
define arcsin ( x )  {  return 2 * arctan ( x / ( 1 + sqrt ( 1-x ^ 2 ))) ; }
define arccos ( x )  {  return pi / 2 - arcsin ( x ) ; }
define arccot ( x )  {  return pi / 2 - arctan ( x ) ; }
define arccsc ( x )  {  return arcsin ( 1 / x ) ; }
define arcsec ( x )  {  return arccos ( 1 / x ) ; }

```

**This time we have a full trigonometric library and we will use it -

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

**In this way, we can use the file to build our own libraries. 
But after all, you feel that what is the use of doing all these things if we can not properly calculate all the values. Well you may not return proper output for all the input, but you can return proper output for some special inputs, such as cos (pi / 2), sin (pi / 2) etc. You can return the correct value for these special input parameter values. For that you have to learn a little more programming. Let's see how we can return proper output for a particular input. 
To accommodate this change, I've changed the previous file -

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

```

```

Now, see I used if for every special input, it is actually a keyword that helps **bc to indicate that we are checking for a particular situation, and if that particular situation arises, we will run the program in one way, otherwise run the program in another way. We will write this special situations in parentheses. In case of special circumstances such as tan, when we input pi/2, we try to return infinity (in fact infinity is not possible in real life, so we're returning 10^40, considering it large enough for our calculation); in another case we are returning **sin(x)/cos(x). In the same way sin, cos, cot, sec, cosec each have its own special input values where we need special return rather than the calculated output. We will now calculate some values ​​by running bc -

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

**But after that also we can see that the value of tan(pi/4 ) or  cot(pi/4) is not accurate, so we will change the file again -

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

**In this case, we  used the else if to indicate two different situations . That means if the situation of if is not met,  else if will be checked, or it will run the else part of the program. In  case of **tan**(x**), if input is pi/2 then it will return infinity, for pi/4 it will return 1 and in other cases it will return the value according to the formula. Let's see once -

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
sin(radian(90))
.9999998001333682524818408858972908885418

cos(radian(90))
-.0006322445915532736545183507752145567248

tan(radian(90))
-1581.6660411069837079729290816982181726254171
```

**You may have noticed the last few lines. Maybe many people have understood the reason why the discrepancy. Actually **degree*((22/7)/180)is not exactly same as its radian equivalent, so tan function can not run the correct segment of program and returns incorrect output.
Let's see how it can be fixed. Here we will take the help of another operator, which is called **boolean or operator. If you do not know about boolean algebra, then I will embed a video here that will help you understand -
&lt;/iframe&gt;

This time we will see how we will use this **boolean or operation in **bc. For that I'll make a little change in the previous program -

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

**With this input to this modified program we will calculate some values ​​again -

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

**In this way you can do various types of calculations by programming and using different operators. In our next post we will see how we can calculate the factorial with the help of loop.

<div class='
