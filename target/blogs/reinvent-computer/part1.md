# Reinventing the Computer

Over the last few years, I’ve been diving into different machine learning algorithms, data storage systems and distributed systems. While exploring, one question caught me off guard:

> How can a machine that only understands ON/OFF signals actually collaborate with humans and join a conversation?

This question sent me back to my college days, when we crammed decades of computational research into four short years — and, as often happens, promptly forgot most of it after graduation. So, I decided to start fresh: to research the topic from the ground up and document the journey.

This series will take us from the very first building blocks of computation all the way to the basics of machine learning.

---

## Back to the Beginning

Imagine it’s the era when transistors have just been invented. Boolean algebra has been formalized, binary numbers are here to stay, and the basic logic of computation is ready to be put to use. Work has already been done with vacuum tubes, the Universal Turing Machine has been conceived, and the 2’s complement system is being developed.

With just a handful of transistors, we can create **logic gates**. For example, an **AND** gate works like this:

```
0 AND 0 = 0
0 AND 1 = 0
1 AND 0 = 0
1 AND 1 = 1
```

Similarly, a **NOT** gate flips a signal:

```
NOT 0 = 1
NOT 1 = 0
```

We can package these tiny circuits as reusable building blocks — starting with **AND** and **NOT** gates.

---

## Counting in Binary

Once we have logic, we can move on to **binary counting** — the foundation of all digital math.

```
0 + 0 = 0
0 + 1 = 1
```

So far, so good. But to add these, we need an **OR** gate — and we don’t have one yet.

Boolean logic tells us:

```
A OR B = NOT (NOT A AND NOT B)
```

Using just AND and NOT gates, we can build an OR gate. Let’s check it:

| A | B | NOT A | NOT B | NOT(NOT A AND NOT B) |
| - | - | ----- | ----- | -------------------- |
| 0 | 0 | 1     | 1     | 0                    |
| 0 | 1 | 1     | 0     | 1                    |
| 1 | 0 | 0     | 1     | 1                    |
| 1 | 1 | 0     | 0     | 1                    |

It works! We’ve just built our first new logic gate from the ones we already had — a small step, but a key idea in computer design.

---

## From Counting to Adding

So far, we’ve counted in binary: just `0` and `1`.

But what can we actually *do* with these numbers? Let’s try adding them:

```
0 + 0 = 0
0 + 1 = 1
1 + 1 = 10
```

Wait… what just happened with `1 + 1`?

For the first time, we see that adding two inputs can produce *two* outputs. In binary, this is normal — just like in decimal when you add `1` to `9` and get a carry of `1` and a sum of `0`. Here, adding `1` and `1` gives us a carry of `1` and a sum of `0`.

But there’s a challenge: with only one logic gate, we can’t produce both outputs. We’ll need **two** gates — one for the sum and one for the carry. And if we have two gates, we’ll need to handle two inputs for each case.

The fix is simple: pad with a leading zero, just like `09` in decimal still means `9`. In binary, `01` still means `1`.

---

## Writing Out All the Cases

Let’s list the possibilities:

```
00 + 00 = 00
00 + 01 = 01
01 + 01 = 10
10 + 01 = 11
```

Now we can separate the addition into two problems — one circuit for the **sum bit**, and one for the **carry bit**.

**Sum:**

```
0 + 0 = 0
0 + 1 = 1
1 + 0 = 1
1 + 1 = 0
```

**Carry:**

```
0 + 0 = 0
0 + 1 = 0
1 + 0 = 0
1 + 1 = 1
```

The carry is easy — that’s just an **AND gate**.

The sum is trickier. We need a circuit that outputs `1` when the inputs are *different*, and `0` when they’re the same. That logic looks like this:

```
SUM(A, B) = (A AND NOT B) OR (B AND NOT A)
```

Let’s check it:

| A | B | SUM |
| - | - | --- |
| 0 | 0 | 0   |
| 0 | 1 | 1   |
| 1 | 0 | 1   |
| 1 | 1 | 0   |

Perfect. We’ve just created a new logic gate: **XOR** — “exclusive OR.”

