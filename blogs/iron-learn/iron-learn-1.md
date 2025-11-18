# Resuming my journey on learning the basics of AI
I have attempted to learn Rust several times in the past. The journey always looked the same: start learning an area of development, implement a few things, understand the concept; and then leave it after some time.

This happened more than once, so I decided to try a different approach this time.

I was also curious about how AI works. I don't quite like the "magic" of black boxes and machine learning felt like one to me. So, I set out on a new journey: to merge both of my areas of learning in one project. I decided to write a Rust-based ML library.

It went well. I spent about a month to get to the point where it can do simple linear regression. But then, due to other priorities, the project went on the back burner, collecting dust.

After about 18 months, I found some free time to pursue the library once again and decided to document my journey this time.

And that’s where the story begins...

## Clearing up the dust
The first thing that came to my notice was that I was able to run my program with some synthetic data. To validate the library, what I did was very basic in nature. I set up an equation in the form of `y = mx + c` and made up some random points for `x` and added a constant `c`. I calculated the equation and added a small random noise to generate 1000 data points to feed into my library.

In this setup, the library was giving a near-perfect answer and I was very happy about it.

I checked the inventory, here it was: 
1. A `Tensor` library to perform the math
1. A `Gradient Descent` function to converge to the solution

It was good enough for the trivial use case. I connected all the dots and moved on to the next step.

## Starting with real data
The solution was working on a synthetic dataset. Hence, I decided to give it a spin with a real dataset. I searched Kaggle. After checking a few options, I picked one with 414 data records, divided it into train and test sets: 321 and 93. I started my test on the library.

The rabbit hole opened up...

### Journey with `NaN`
The first result that I was able to generate with my `prediction` was full of `NaN` values. I was not at all surprised to see that. I tested it against a trivial, made-up dataset. Things were bound to break in a real-world scenario.

I rolled up my sleeves and started the debugging process. I put a lot of `println!` statements in every block of code. The result came faster than expected. At some point, the multiplication was returning numbers so large that they produced `NaN`. My tensors were using floating-point numbers and they were capable of producing `NaN`.

I started reading about the issue and took some help from Copilot to understand that the step in the process I was missing was the normalization of data. I implemented a normalization method and invoked it before feeding the data into the linear regression module.

After this, I was able to see some real numbers instead of `NaN`.

Phew!!!

### The realization and validation
After fixing the `NaN`, my next validation was to check if the returned results were correct or not. Well, a 70% success rate. Not bad but still felt something amiss. Curious, I opened a python shell this time to conjure a sklearn-based linear regression program and fed the same data.

As expected, the python program returned a much lower error, around 19% (needless to say the speed difference which I deliberately ignored). I was back on the path of debugging to find out what needs to change for better accuracy.

### The fix
I looked at each line of the code and tried to understand, where I was going wrong. No error caught my eyes. However, I also noticed that my learning rate was very small - 0.000015 with a very low iteration - 1000.

I bumped up the learning rate, increased it to *0.001*.

Voila, my program was producing almost similar results to that of sklearn's.

### Final Results

```shell
Sklearn
=================
Results:
Total test samples: 93
Mean Squared Error: 59.6778
Root MSE: 7.7251

My program
=================
Results:
Total test samples: 93
Mean Squared Error: 60.1019
Root MSE: 7.7525
```




