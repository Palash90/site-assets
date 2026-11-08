---
title: "Another step in the command line - arithmetic"
date: "2017-07-16T12:25:00+05:30"
recovered_from: "web.archive.org (snapshot 20190917002736)"
---

After running some commands in the command line, you are now confident, at least now the command line or shell does not bother you much. Now we'll forward one more step in our journey. We will now learn something which is less likely to mess the operating system.

We'll learn to calculate in the command line. Generally, any operating system now has a graphical calculator installed, so most people do not show any enthusiasm about command line calculators. But from my experience, I can say that it is proven to be quite useful and it is irritating to use the graphical calculator while working on the command line, so at least I think it is better to know about the command line calculator. Moreover, the simplest way to get used to the command line is to  use command line calculator. 
Generally, all Unix / Linux based operating systems have bc by default, so we will use it for our work. Once you are used to it, you can choose another command line calculator. Let's see how it looks like -
![image](https://2.bp.blogspot.com/-myNp1BH4j1k/WWT9uA4NBYI/AAAAAAAAG2w/c5sL5sLI5BMi_Jgl3m4BMwwfsnOZwwy8ACLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-11%2B22-02-37.png)

When bc is run, it often provides information about itself, sometimes it does not provide any information and only shows the prompt . We will now see how to add subtract multiply and divide -

```
totan@home-computer~$ bc
Bc 1.06.95
Copyright 1991-1994, 1997, 1998, 2000, 2004, 2006 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
Details For type  ` warranty ' .
78
78
98 + 75
173
45-3
42
22.12 + 14.98
37.10
22.47-5.6
16.87
7 * 5
35
3/2
1

```

**If you look carefully, the first thing that you can see is the pattern of input / output. In the first line of bc you will give your input and bc will show you the output on the next line . You can see that adding and subtracting is very easy. But if you are not an existing computer user, then multiplication symbol might look a little weird to you. In fact, this symbol is called asterisk, and it is used in mathematical terms as multiplication symbol. 
Everything seems perfect except the division, what happened to it ? 

While performing 3/2, shell seemed to be wrong. What a bad calculator!

Actually, shell did just what we asked it to do. In fact, shell showed us the whole part of the division result (quotient) but it did not show the decimal part. If we want to know the remainder, then we have to use % -

```
3%2
1

```

**In this case bc does not show the quotient, just the remainder. That means % performs the modulus operation. We will try to understand the matter with a few more examples -

```
10/3
3

10%3
1

12/5
2

12%5
2

12/6
2

12%6
0

13/6
2

13%6
1

15/7
2

15%7
1

```

**10 divided by 3 quotient is 3 and the remainder is 1; 10/3 and 10% 3 respectively show the value 
12 divided by 5 quotient is 2 and the remainder is 2; 12/5 and 12% 5 respectively show the value 
12 divided by 6 quotient is 2 and the remainder is 0; 12/6 and 12% 6 respectively show that value 
divided by 13 is 6 quotient is 2 and the remainder is 1; 13/6 and 13% 6 respectively showed that value 
divided by 15 to 7 quotient is 2 and the remainder is 1; 15/7 and 15% 7 showed that value as well
**One more thing to notice, to keep the screen clean and easy to read, I have now pressed Ctrl - m or Enter after each calculation, which means there is one blank line in between the calculations for better readability. 
**Many people seemingly peek at this question, can't we divide in fractions ? 
Sure you can do that, you just have to tell shell the number of decimal places you want to see in the fraction part. For that you have to write a small program, much like this -

```
Scale = 3;

3/2
1.500

4/3
1.333

10/9
1.111

3/7
.428

22/7
3.142

2/3
.666

6/9
.666

Scale = 4;

4/3
1.3333

2/3
.6666

10/9
1.1111

3/7
.4285

Scale = 34;

22/7
3.1428571428571428571428571428571428

4/3
1.3333333333333333333333333333333333

10/9
1.1111111111111111111111111111111111

```

**Before throwing all your queries at me, congratulate yourself as you have written the first program in Shell. 
Now I'll answer your question, yes you wrote a very short one-line program and not once, many times (to be exact, 3 times). Well, you probably did not notice in the excitement of using bc, you wrote  scale = 3; in the shell and shell answered you up to 3 decimal places, similarly you wrote scale = 4; And  scale = 34; and shell showed upto 4 and 34 decimal places. In fact when bc is run the value of scale is 0 or programmatically we can say scale = 0; so, when 3/2 is run, it does not show any answer to decimal places. So we wrote the program and changed the value of scale and shell showed us results upto that much decimal place. The same way you can change the scale to the different decimal places, see the fraction or see the quotient and remainder in further calculations.

Now is the time for some shortcuts. You can see the previous commands as you press the up / down arrow and you can run it again as needed, much like you did in shell. But remember by typing exit, you can not get out of bc; to exit from bc you have to type quit and press Ctrl - m. Another way to exit the bc is  press Ctrl-d. Ctrl-d indicates end of input or no input is left, so bc is turned off. Other than bc,  Ctrl-d is also used in another shell commands, we will discuss about them later. By using bc as your calculator, you will gradually be getting used to the command line. So, we will see, what else is possible with the help of bc .

<div class='
