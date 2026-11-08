---
title: "bc - logarithm, exponential and function"
date: "2016-07-17T12:28:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045016)"
---

We have seen how to change a number to a different base, we've changed some numbers from one base to another. We have seen how we can perform various trigonometric operations. We have seen how to calculate different trigonometric values ​​with the help of a single trigonometric operation and identity. Now we will know how to calculate the value of logarithm. Again, I've to say that **bc can only evaluate the value of natural logarithm or **ln, we will use the help of identities for determining the logarithm of another base, so we will keep some formulas in mind -

log  y  (x) = ln (x) / ln (y)

ln (e ^ x) = x 

e ^ ln (x) = x

Now we will look at some examples on how to calculate logarithm and exponential using **bc -

```
Totan @ home-computer ~ $ bc -l
Bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
Details For type  ` warranty ' .

x = 1

e ( x )
2.71828182845904523536

l ( x )
0

l ( 10 )
2.30258509299404568401

l ( x ) / l ( 10 )
0

x = 100

l ( x ) / l ( 10 )
2.00000000000000000000

x = 100000000
l ( x ) / l ( 10 )
8.00000000000000000002

x = 15

e ( l ( x ))
14.99999999999999999990

l ( e ( x ))
14.99999999999999999999

l ( x )
2.70805020110221006599

x = 16

l ( x )
2.77258872223978123766

l ( x ) / l ( 2 )
4.00000000000000000002

l ( x ) / l ( 4 )
2.00000000000000000000

```

If we want to run a formula repeatedly for different values, we have to repeatedly change the value of the variable (in this case x) and then run the formula again. It may not be very difficult for small calculations, but it is very difficult for bigger formulas. So we will learn a new way to solve such problems and we'll go one step further in programming, and will see how to write functions in **bc. In simple language function is a programming block, where we write together one or more line of programs and name it according to our choice and later provide input to the function to get result. Now we will see how to write the function and provide input to it.

```
define log ( x ) {  return l ( x ) / l ( 10 ) ; }

log ( 10000 )
4.00000000000000000001

define log ( x, y ) {  return l ( y ) / l ( x ) ; }

log ( 2, 8 )
3.00000000000000000002

log ( 625, 5 )
.25000000000000000000

log ( 1024, 2 )
.09999999999999999999

log ( 2, 1024 )
10.00000000000000000010

log ( 5, 625 )
4.00000000000000000000

log ( 10 )
Runtime error ( func = ( main ) , adr = 8 ) : Parameter number match

```

Reading the code carefully, we can understand what the function is doing. In the first case, it evaluates **log base 10 value. After that, we change the function to accept more input value and we are computing the value of log x (y) in the changed function . 
Now notice the last line, when we change the function, and we provide it only one input value, it does not work, because now the function is taking two values ​​as input. These input values ​​are called parameters in programming language. Now think about how beautiful the trigonometric formulas could be with the use of functions. I will not do them for you. This work is for you, changing the trigonometric formulas in a well-formed function and use them. 

After this, it is easy to work but you will notice that when you come out of **bc and type **bc again, you will lose the written functions of the previous session. So, you really want to save all your work and use them later. 

But for that you have to know some editor and file system. If you have any idea about these, then you can proceed in the next section - [bc - files and programming](http://unix.palash90.in/2016/07/bc-files-and-programming.html). Otherwise, check for some file editor and file system details of Unix/Linux

<div class='
