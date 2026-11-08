---
title: "bc - scientific calculation"
date: "2016-07-17T12:29:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021051001)"
---

**When writing this post, I will assume that - 
1 You can do scientific operation 
2. You are aware of bc, if not, if you can see this post - [bc](http://unix.palash90.in/2017/07/another-step-in-command-line-arithmetic.html)
**We will learn our next arithmetic operation. With this operation, we can change any number from one base to another. For this, we will use the other two variables obase and ibase like scale . If you do not know what is a variable, keep in mind, it's a placeholder whose value can be changed and while bc calculates it shows the answer based on the values ​​of these variables. Variables are important for programming and if you want to continue your studies with bc, you have to learn bc programming eventually , so gradually knowing these small parts is good. obase is used to indicate in which base output will be shown. The same way ibase is used to indicate in which base the input is being provided to bc. When bc is run, the value of both ibase and obase is 10. ibase take values ​​from 2 to 16 and obase from 2 to 999. But for day to day work, 2 to 16 is enough, and here also we will calculate using the base from 2 to 16. Let's try to understand the matter with a few examples -
**

```
ibase = 2

0
0

1
1

10
2

11
3

100
4

101
5

110
6

111
7

78
3

79
3

11
3

70
2

90
2

10
2

01
1

45
3

11
3

61
3

111
7

789
7

654
7

789
7

```

**Or

```
ibase
3

obase
10

10
3

11
4

12
5

13
5

14
5

15
5

21
7

10
3

01
1

31
7

21
7

```

**There are two things to note here, when we type ibase or obase , we see one value, actually these are the value of those variables which means, the input in the second case is given at 3 base and the output is shown at 10 base. One more thing is to be noticed, in both cases, when we input more than the digits of a base, it takes the highest value digit of that base. When ibase was 3 then15 was 5 (the highest digit is 2 in base 3, so bc took 15 as 12, on base 10 which is 5), same with ibase  when 2 is the base and input was 78 and 79, it was taken as 11, which in base 10 is 3.
**Let's see some examples of obase -

```
obase = 16

ibase
A

obase
10

17
11

15
F

16
10

14
E

125
7D

67
43

546
222

554
22A

668
29C

4854
12F6

ADF
3E7

```

**Do not be worried seeing the first two lines, 16 shows 10 in base 16 and 10 shows A. Here also you can see I gave the input as ADF which took the input as the 999 and showed the hexadecimal 3E7 as answer. 
**This time we will change the two variables together and change from binary to hexadecimal -

```
Totan @ home-computer ~ $ bc
Bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
Details For type  ` warranty ' .

obase = 16
ibase = 2

obase
10

ibase
2

11
3

1111
F

11111111
FF

10101100
AC

1011011010010100101
5B4A5

10101001
A9

010111011010101010010100010100
176AA514

11 + 11
6

1101 + 1101
1A

11 + 101
8

1101 + 101
12

111 + 10
9

```

**Not only changing one number from one base to another, we can add two numbers in a base and change the result to show on another base. We'll take a look at some more examples -

```
Totan @ home-computer ~ $ bc
Bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
Details For type  ` warranty ' .

obase = 16
ibase = 2

obase
10

ibase
2

sqrt ( 100 )
2

sqrt ( 11 )
1

scale = 10

sqrt ( 11 )
1.BA

sqrt ( 1001 )
3.00

sqrt ( 1111 )
3.DE

sqrt ( 1111 )
3.DE

sqrt ( 11111111 )
F.F5

11-1
2

11-11111
-1C

11 * 11
9

10 * 01
2

10/1
2.00

1001/1101
.B0

```

**In addition to base changes, bc does some scientific operations, such as sine, cosine, logarithm, arctangent, exponential etc. 
For this we need to type bc -l in our command line. -l is an option that helps bc to load all operations in mathematical libraries. Some of the operations are in the library -

S (x): The sine of x, x is in radians 

C (x): The cosine of x, x is in radians.

A (x): The arctangent of x, arctangent returns radians.

L (x): The natural logarithm of x.

E (x): The exponential function of raising e to the value x.

J (n, x): The bessel function of integer order n of x

Because bc can not calculate all types of trigonometry operations, so we will know some identities -

tan (x) = sin (x) / cos (x)

cot (x) = cos (x) / sin (x)

Sec (x) = 1 / cos (x)

cosec (x) = 1 / sin (x)

arcsin ( *x ) = 2 * arctan (x / (1 + sqrt (1-x ^ 2)))

arccos ( *x ) = pi / 2 - arcsin ( *x ) = pi / 2 - 2 * arctan (x / (1 + sqrt (1-x ^ 2)))

arccot ​​(x) = pi / 2 - arctan (x) 

arccsc (x) = arcsin (1 / x) = 2 * arctan ((1 / x) / (1 + sqrt (1- (1 / x) ^ 2)))

arcsec (x) = arccos (1 / x) = pi / 2 - arcsin (1 / *x ) = pi / 2 - 2 * arctan ((1 / x) / (1 + sqrt (1- (1 / x) ^ 2)))

degree = radian * (180 / (22/7))

radian = degree * ((22/7) / 180)

With the help of these identities, we can calculate the value of all the trigonometric operations using a (x), s (x), c (x) operation. This time we will take a look at these operations with a few examples -

![image](https://4.bp.blogspot.com/-xbdOjYAM0ZQ/WWbxW3zEmRI/AAAAAAAAG3o/me7uckVfAj0mUeWsqJPzcyKEkV1SJdEKQCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-13%2B09-34-19.png)

Notice a few things -

Now the value of scale is 20. When wee run bc -l, it automatically sets scale to 20 

When we use s, c operations, we change from degree to radian and provide input.

There is no operation to determine tan (x), sec (x), cot (x) and cosec (x), so we can calculate the value of tan, cot, sec and cosec with the help of an identity using s and c.

Valid values ​​are more important for delicate calculations. So you will pay more importance to python, java, matlab, etc. for very accurate calculation than bc.  

Now we will take a look at some examples of arctangent -

```
a ( 9999999999 )
1.57079632669489661922

a ( 9999999999 ) * ( 180 / ( 22/7 ))
89.96378961979862455540

a ( 0 )
0

a ( 1 )
.78539816339744830961

a ( 1 ) * ( 180 / ( 22/7 ))
44.98189481276294864133

```

The first answer we see in the radian unit and we are changing it to degree with the help of a formula. 
With the help of a(x), I will find out the value of arcsin -

```
x = 1

2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))
1.57079632679489661922

2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 ))) * ( 180 / ( 22/7 ))
89.96378962552589728267
A ( 1 )
.78539816339744830961

a ( 1 ) * ( 180 / ( 22/7 ))
44.98189481276294864133

x = 0.5

2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))
.52359877559829887306

2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 ))) * ( 180 / ( 22/7 ))
29.98792987517529909346

x = 0

2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))
0

2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 ))) * ( 180 / ( 22/7 ))
0

```

Look at one thing, we're giving value of x like scale or obase or ibase. That means we're using x as a variable. In this way we can write complex formulas easily and can run repeatedly and by giving different values ​​of x, we can apply the same formula over different input values. 

Now we will calculate arccos (x) values -

```
x = 1

(( 22/7 ) / 2 ) - 2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))
.00063224463367480935

((( 22/7 ) / 2 ) - 2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))) * ( 180 / ( 22/7 ))
.03621037447410271731

x = 0

(( 22/7 ) / 2 ) - 2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))
1.57142857142857142857

((( 22/7 ) / 2 ) - 2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))) * ( 180 / ( 22/7 ))
89.99999999999999999999

x = 0.5

(( 22/7 ) / 2 ) - 2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))
1.04782979583027255551

((( 22/7 ) / 2 ) - 2 * a ( x / ( 1 + sqrt ( 1-x ^ 2 )))) * ( 180 / ( 22/7 ))
60.01207012482470090653

```

This time we will see some arccot's value--

```
x = 1

(( 22/7 ) / 2 ) - a ( x )
.78603040803112311896

((( 22/7 ) / 2 ) - a ( x ) * ( 180 / ( 22/7 ))
45.01810518723705135865

x = 0

(( 22/7 ) / 2 ) - a ( x )
1.57142857142857142857

((( 22/7 ) / 2 ) - a ( x ) * ( 180 / ( 22/7 ))
89.99999999999999999999

```

Now we will see how we can calculate the value of arcsec with the help of identity -

```
x = 2

2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))
.52359877559829887306

( 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))) * ( 180 / ( 22/7 ))
29.98792987517529909346

x = 1

2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))
1.57079632679489661922

( 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))) * ( 180 / ( 22/7 ))
89.96378962552589728267

x = 4

2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))
.25268025514207865348

( 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))) * ( 180 / ( 22/7 ))
14.47168733995541379023

```

Finally we conclude this post by evaluating some of the arcsec values ​​-

```
pi = 22/7

x = 99999999999999999

pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))
1.57142857142857141857

( pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))) * ( 180 / ( 22/7 ))
89.99999999999999942726

x = 4

pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))
1.31874831628649277509

( pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))) * ( 180 / ( 22/7 ))
75.52831266004458620976

x = 0.2

pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 ))
Runtime error ( func = ( main ) , adr = 26 ) : Square root of a negative Number

x = 5

pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))
1.37007065063824063713

( pi / 2 - 2 * a (( 1 / x ) / ( 1 + sqrt ( 1- ( 1 / x ) ^ 2 )))) * ( 180 / ( 22/7 ))
78.46768271837196376296

```

Here you will notice two things. I will not tell you anything about them, think about it and try to find answers.
Next we'll see some more scientific calculations.

<div class='
