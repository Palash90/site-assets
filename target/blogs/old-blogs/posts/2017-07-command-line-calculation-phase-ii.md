---
title: "Command Line Calculation - Phase II"
date: "2017-07-16T12:24:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045011)"
---

In our last post, we have seen how to add, subtract, multiply and divide numbers in command line, and now one step further, here we will learn how to make another type of calculation in command line. 
**Let us now see how we can evaluate the square root of a number in the command line -
![image](https://4.bp.blogspot.com/-a8uyUZNRng0/WWY87ouzONI/AAAAAAAAG3Q/FTpsLE2m_JYukB4fDhTfkbGHuEflO-OjACLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-12%2B20-44-33.png)
**Again wrong answer? 
Think a little 
**This time you've to understand what you have to do. If you do not understand, then think before reading the next line, think about whether you can remember something. 
**Well, many people have understood this, maybe some have even taken the right path to see the exact answer in command line. 
We have to provide the value of scale -

```
Scale = 30
Sqrt ( 2 )
1.414213562373095048801688724209

```

**You may have noticed, I have removed the ; after  scale = 30 and still it works, so ; is optional in bc. 
**Now we will take a look on how we can evaluate the value of a number raised to different power -

```
2^2
4

1^1
1

3^3
27

4^4
256

6^6
46656

6^2
36

9.2^2
84.6

15.7^2
246.4

15.7*15.7
246.4

Scale=2;

15.7*15.7
246.49

15.7^2
246.49

15.7^3
3869.89

Scale=3;

15.7^3
3869.893

```

**At this time you have surely understood that by setting different values ​​of scale, we are seeing the answers in different decimal places. 
**Many times it may happen that, shell does not show the proper result as what we're trying to calculate. Shell is actually calculating exactly what it is supposed to do but we just don't know how to convey it properly as the below example illustrates -

```
3^3^3
7625597484987

```

**After seeing this answer, you might think, what happened, I thought 3 ^ 3 ^ 3 would be 19683, but why did shell show 7625597484987?
Actually no mistake from shell, shell just did what it is supposed to do. But there is a little lack of synchronisation between you and shell. These kinds of problems are often encountered, so we will see how to get rid of such problems -

```
3 ^ ( 3 ^ 3 )
7625597484987

( 3 ^ 3 ) ^ 3
19683

```

**In the first case it is being seen as - 3 ^ ( 3 ^ 3 ) = 3 ^ (27) =  7625597484987 
and in the second case -  ( 3 ^ 3 ) ^ 3 = 27 ^ 3 = 19683. That is, by using parentheses, we change the sequence of execution. If you remove parentheses, both are essentially the same 3^3^3, isn't it ?
**Let's look at some more and it will be clear to us -

```
Scale = 5;

7 + 6 * 5
37

7+ ( 6 * 5 )
37

( 7 + 6 ) * 5
65

1/2 + 8 * 9
72.50000

1 / ( 2 + 8 ) * 9
.90000

1 / ( 2 + 8 * 9 )
.01351

```

**By using braces in this way we can accomplish a bigger and more complex calculation in one step. With some examples we can see -

```
7 + 6 * ( 5 + 5 * ( 12 ^ 3 ) + 19 * 5 ^ ( 2 * 3 ) + 6 * ( 3 ^ 2 ) ^ 9 ) / 9 + 78
1549885721

( 7 * 9 ) + 2.2 ^ 3 + 4.5 * 8 * ( 8 + 9.2 ^ 2 + 5 ) + 10 * 10 + 91 + 2 ^ 4
3795.688

```

**So far we have come to know the method of basic calculations of all kinds. Hopefully, now your fear about the Command Line have been cut short and you can work on command line more comfortably than before. We will go back to shell again to learn other commands in the Command Line. 
**For those who liked bc and want to know more about bc, I have described the method of calculating some more scientific stuffs in a mini series, willing readers can head to - [bc - scientific calculation](http://unix.palash90.in/2016/07/bc-scientific-calculation.html)

<div class='
