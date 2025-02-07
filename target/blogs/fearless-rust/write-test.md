Fearless Rust
=============
> “The more you sweat in peace, the less you bleed in battle.” - Norman Schwarzkopf

Rust is notorious for it's learning curve and this is the main reason many people start learning rust and leave in between. However, if you stick to it, the concepts are not hard to understand. If you follow the Rust rules, you are going to think better solutions to a problem. Rust compiler is tough to ensure that you don't hammer on your own feet.

That's why, I am trying to practice this language more and more to face more and more compiler errors in toy problems and eventually solve them. So that, I can be prepared to handle more practical and real life solutions.

Test Driven Development
=======================
I have been in industry for more than 13 years now. One thing I have seen for sure is that, almost 60% of the QA Identified bugs are coming from developer mistakes.

While working with a development project in GE, I was assigned a lot of times, silly bugs which was due to my silly mistakes and sometimes, just because, I did not have time to verify my changes. This was the time, I genuinely thought of taking a different approach. Talked to my lead and my manager about my decision of trying TDD or Test Driven Development.

They both were kind enough to listen to me and brave enough to spend extra time on TDD during a time crunch situation. What I found was astonishing to me and to others. When I started, I was not "productive" at all for straight 6 weeks because, I was not productive or delivering anything. This 6 weeks, I was trying many versions of test inputs and outputs which can break or make the product. This was the toughest time to justify. In fact, I tailor made a simple library to suit our project by leveraging MS Test. Then after 6 whole weeks, I actually started on the real solution. This took me around 3 days to complete and satisfy all test cases.

When I got another task of similar type, it took just one single week to complete. The library was ready to test, the process of finding out fallible tests were known. It was just a matter of writing down my thoughts in computer language.

The end result was even more astonishing. Both the modules I wrote passed QA with a negligible number of bugs and it ran in production for more than 1 year without any issues.

That's why the very first program, we'll write is a test. And we'll follow this practice going forward. We'll first understand the scenario and then brainstorm on the potential failures and then only we'll find the solution to the problem and then finally will write the code.

Similarity of TDD in Rust Development
=====================================
The Rust compiler is a tough one to satisfy. There are so many checks around the resource usage built in the language itself that it is definitely hard to go wrong in the run time. However, make no mistake that bugs still follow. But it is hard to come by when you are writing a Rust Program.

I draw the analogy of tough compilation rules of Rust to TDD. They may be hard to understand and conceive but once you do, your path is easier ahead.

Memory bugs are definitely the most notorious category of bugs and have the potential to take life or crash a rocket. Rust tries most of this category bugs untill you are doing something `unsafe`. We will talk more about all these in due course of time.

The Setup
=========
If you have already setup a development environment for Rust, you can freely skip this section. For everyone else, we need the basics.

1. Some concept of programming (not necessarily low level programming)
1. Rust tool chain. For this, you have to head over to [Rust Website](https://www.rust-lang.org/) and follow the instructions of installation
1. VS Code or any favourite IDE
1. Rust-Analyzer extension for VS Code

These are all the basics. If you are a new comer to Rust, stick to the series and you will learn the language by doing things. You should practice along with reading this blog.

However, if you have gone through [The Book](https://doc.rust-lang.org/book/) or learnt Rust from some other resource and wanted to know what next? then this blog/video series is definitely worth considering.

The Hands On
============
Once you have installed Rust tools and setup VS Code, you will want to start your journey. To do this, the very first thing is that, you want to start a library project. To do that, you create any folder or directory in your system and type the following in the terminal.

```
cargo init --lib .
```

This will create a new library project with all the basic boilerplate like the following.

```
src
    lib.rs
Cargo.toml
```
There are many more, which we don't need to know as of now. These are the two main files that are of our immediate interest.

The `src` directory is where all your source codes will reside and `Cargo.toml` is there to help the cargo tool chain to understand your project and take necessary action when you key in `cargo run` or `cargo test`.

In the `lib.rs` file, you will find the following code already written for you.

```
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
```

As you might have noticed, the file already contains the test file and the corresponding function. Now, let's run the test using the following command.

```
cargo test
```
As you might expect, the tests run well and passes everything. However, test cycles are not always Green and for TDD, we should first see the Red then the Green. So, now add some red to the mix. Let's  add one more simple test.

```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn sub_works() {
        let result = sub(2, 2);
        assert_eq!(0, result);
    }
}
```
As you can now see we added one more test for subtraction function. Now let's run the test again.

```
fearless-rust> cargo test
   Compiling fearless-rust v0.1.0 (C:\Users\iampk\git\personal\fearless-rust)
error[E0425]: cannot find function `sub` in this scope
  --> src\lib.rs:17:22
   |
17 |         let result = sub(2, 2);
   |                      ^^^ not found in this scope
   |
help: use the `.` operator to call the method `Sub::sub` on `{integer}`
   |
17 -         let result = sub(2, 2);
17 +         let result = 2.sub(2);
   |

For more information about this error, try `rustc --explain E0425`.
error: could not compile `fearless-rust` (lib test) due to 1 previous error
fearless-rust> 
```
Now you see that the compilation is breaking as we don't have `sub` function. The compiler also tries to help with an alternative like using the `.` operator. The helpful compiler tries to help and many a times you will find that the compiler is giving you exact solutions to the problem compiler found. However, for this time, we are ignoring the compiler and adding the `sub` function ourselves. Let's add the `sub` function to the file.

```
pub fn sub(left: u64, right: u64) -> u64 {
    left - right
}
```
You can place this function anywhere in the file and run the test and you will find that again both the tests are passing.

```
fearless-rust> cargo test
   Compiling fearless-rust v0.1.0 (C:\Users\iampk\git\personal\fearless-rust)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.44s
     Running unittests src\lib.rs (target\debug\deps\fearless_rust-e9c43e60644089ef.exe)

running 2 tests
test tests::it_works ... ok
test tests::sub_works ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests fearless_rust

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

fearless-rust> 
```

So, that's all for this post. Tinker as much as you can. May be add some more tests with negative numbers. Or try to change the data type if necessary. Tinker around and see what happens.
