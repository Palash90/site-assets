---
title: "Getting Started with Command Line"
date: "2017-07-16T12:26:00+05:30"
recovered_from: "web.archive.org (snapshot 20190917002843)"
---

Hopefully it's definitely clear to you, what Unix and Linux is and how today it is. At the same time you can have an idea of the terminal, console and shell etc. So you understand what the command line is or what it does. Now, we will start the command line and type some commands to see how the command line works. I assume that you are brand new to the shell or terminal, so maybe few start-up posts will be at the preliminary level; If you have experience with Unix, Linux or Shell, you might think some stuffs very obvious and it needs no mention.

Typically , the terminal emulator is opened by pressing these three keys together in Linux Mint or Ubuntu: Ctrl-Alt -t, on other Linux-based systems, the Terminal Emulator can be opened from the System menu. Below is a picture of my current terminal -
![image](https://2.bp.blogspot.com/-XtrJWSvsKj0/WWDTlraoRwI/AAAAAAAAG1I/i4eRN3meM2kJ8z6wAIa9HsN8luqHMoJkACLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-08%2B18-13-38.png)

Terminal usually provides some information, such as the user who has logged in, the computer name where the user has logged in, in which directory. After that there is a $ , % or # sign. Generally, most Linux has GNU BASH default, so it is possible that you will also see $ . If a root user logs in, then the # symbol is seen. For example -
![image](https://4.bp.blogspot.com/-nwEKQlc-8gc/WWDVmekGF8I/AAAAAAAAG1M/w2_wzl2zUtk-tneMRNdEtpekLRFnwNU_gCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-08%2B18-22-28.png)

Generally, the root user logs in for administrative activities and the root user is unnecessary for day-to-day activities, and as far as possible , it is best to avoid root user login, we will gradually learn about this. 

When you see $ or % or #, then you understand that Linux is waiting for your command. Generally these symbols are called prompt. Now we will see how to run commands. 
We'll start with the most common command date that shows us today's date. By typing date in the terminal and pressing Enter  , we will see the output. Typically for all commands user has to type it and then press enter to see the output.

```
Totan @ Home-Computer ~ $ date
Sat 8 Jul 18:33:28 IST 2017
totan @ Home-Computer ~ $

```

The next command cal  will show our current calendar month,

```
Totan @ Home-Computer ~ $ cal
July 2017
Su Mo Tu We Th Fr Sa
1
2  3  4  5  6  7  8
9 10 11 12 13 14 15
16 17 18 19 20 21 22
23 24 25 26 27 28 29
30 31
totan @ Home-Computer ~ $

```

Generally, the current day is highlighted, you can understand from the following picture - 
![image](https://3.bp.blogspot.com/-vfDE66WaV-A/WWDZ9zg2ALI/AAAAAAAAG1U/iNzibr5HsAUch6LLDSeFJZxGkzcAXfGVgCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-08%2B18-41-04.png)

If you want to know which users have logged in at the same time, then you have to enter the command who

```
Totan @ home-computer ~ $ who
totan tty8 2017-07-08 17:55 (: 0)
totan @ Home-computer ~ $

```

For example, my system now only has one user logged in, so the current user is showing in the system along with information like which terminal, at what time etc. If only you need to know the current user's information, you can use who -u

```
Totan @ home-computer ~ $ who -u
totan tty8 2017-07-08 17:55 01:13 1864 (: 0)
totan @ Home-computer ~ $

```

Since only one user logged in on my system, who and who -u are showing the same information. 
Now if we want to know which shell program you are using, then use echo $0

```
Totan @ home-computer ~ $ echo $0
bash
totem @ home-computer ~ $

```

My system is using bash, if another shell is used, then we could see the shell program name. For example -

```
$  echo  $0
sh
$

```

```
Home-Computer: ~%  echo  $0
csh
Home-Computer: ~%

```

```
Home-Computer: ~> echo  $ 0
tcsh
Home-Computer: ~>

```

```
Totan @ home-computer ~% echo  $ 0
zsh
Totan @ home-computer ~%

```

```
$ Echo  $ 0
mksh
$

```

**Each shell has some features, so users usually develops preference on one of the available shells. However, since the majority of Linux uses GNU BASH as the default shell, BASH is very popular. 
**In this way, after running some commands, the screen is usually filled up, the old commands go out of the screen and the new commands start in the last of the screen -
![image](https://4.bp.blogspot.com/-1v705sboICI/WWHRnOSDg0I/AAAAAAAAG14/rntPQByYkREaHxzjykwvW-iwF_kNNEubwCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-09%2B12-17-11.png)
**To solve this problem, you can use the clear command so that you can see the cleaner screen omitting all the old commands, where again you can run new commands -

![image](https://2.bp.blogspot.com/-zAo6ht1mKYM/WWHSmzqGHMI/AAAAAAAAG2A/nDjkuhJlLZkO3VEo-1wKJmo0bDsSKDxtQCEwYBhgL/s1600/Screenshot%2Bfrom%2B2017-07-09%2B12-21-56.png)
![image](https://1.bp.blogspot.com/-N3Vz1fwOllk/WWHTDJVS4LI/AAAAAAAAG2E/SmVuO47YGrIqbZXd9t2aHnkjCzq7REi_gCLcBGAs/s1600/Screenshot%2Bfrom%2B2017-07-09%2B12-23-46.png)

This way , after clearing the screen you will not see the old command. 
**Now let's see, what happens when a command can not be recognized by the Linux shell -

```
Totan @ home-computer ~ $ ckwar
Ckwar: command not found
Totan @ home-computer ~ $

```

**When there is some wrong command entry in Linux Shell, there is a message in shell that the entered command is wrong and in many cases the shell can give you some advice, for example, we have run datw incorrectly while typing date command -

```
Totan @ home-computer ~ $ datw
No command  'datw' found, did you mean:
Command 'dat' from package 'liballegro4-dev'  ( universe )
Command 'date' from package 'coreutils'  ( main )
datw: command not found
Totan @ home-computer ~ $

```

**In this case, the terminal advises that you might want to run another command (in this case dat or  date ) but due to typing mistakes you have run another command. In many cases this message can be helpful for typing right command, for example in our case we will run the date command. 
One should always keep in mind that UNIX/Linux is case sensitive, that is, the commands should be small or capital letter. Linux can not recognize it if it runs DATE or date or dAte instead of date, so if you get an error message you should first check the case of the command.

```
Totan @ home-computer ~ $ DATE
DATE: command not found
Totan @ home-computer ~ $ dAte
No command  'dAte' found, did you mean:
Command 'date' from package 'coreutils'  ( main )
dAte: command not found
Totan @ home-computer ~ $ Date
No command  'Date' found, did you mean:
Command 'date' from package 'coreutils'  ( main )
Command 'late' from package 'late'  ( universe )
Command 'kate' from package 'kate'  ( universe )
Command 'yate' from package 'yate'  ( universe )
Date: Command not found
Totan @ home-computer ~ $

```

**Now let's learn some keyboard shortcuts. Linux usually saves the commands you have run. So you can see the previous commands by pressing the up/down arrow to run again the commands that you have run earlier. Now suppose you have to change some command or you have run a wrong command and you have to change it and run again, then press up/down arrow to bring the required command to the prompt and press left/right arrow to make necessary changes and you can execute the command. Usually, all the new keyboards have their own shortcuts, but we can know some shortcut, which is useful at times -

Press Ctrl-M to Enter.

Suppose if you are typing a command and you notice that a wrong character has been pressed somewhere in the command, then you need to press left / right arrow to move the cursor and press backspace or Ctrl - h to remove the character and delete the correct character and press Enter to run the command.

Suppose you typed a big command and maybe typing mistakes are in many places, then you feel like removing the whole line is better than typing, press Ctrl-u to delete the whole line together instead of pressing Ctrl-h repeatedly.

Another way, you can see all the old commands by typing history. history will show you the commands you have run one by one, you can copy/paste as needed . Normally you can select the text with the mouse in the terminal. Usually, if you select something with a mouse in the terminal, the text gets copied and right click will automatically paste the copied text. But if this shortcut does not work, then in most terminal emulators, select a text with the mouse and   press Ctrl - Shift - c , the text is copied and press Ctrl - Shift - V to paste the text into the cursor position.

Daily Practice is the key to overcome the fear of Command Line and to earn skills. So these simple commands need to be run repeatedly, so that the concept of terminal emulator is clear and there is no difficulty in making the necessary changes according to the terminal.
When all your commands are run and you want to close the terminal, you will need to exit the terminal by typing the exit command and pressing Ctrl - m (or Enter).

```
Totan @ home-computer ~ $ exit

```

<div class='
