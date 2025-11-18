# Resuming my journey on learning the basics of AI
I have attempted to learn Rust in past quite a few times. The journey always looked like this - Start learning an area of development, implement a few things, understand the concept and well, leave it at that after some time.

This happened quite a few times. So, thought of a different approach this time.

I was also curious about how AI works. I don't quite like the magic of black box. The Machine Learning was that black box for me. So, I decided on a new journey to merge both of my learning in one project. I decided to write a Rust Based ML Library.

It went cool, I spent about a month to get to the point where it can do simple linear regression. Then due to some other priorities, the project sat alone collecting dust.

After about 18 months, I noticed I got some free time to pursue the library once again and also decided to write the journey this time.

Here we are...

## Clearing up the dust
The very first thing came to my notice is this, I was able to run my program with some made up data. To validate my library, what I did is very basic in nature. I shot up an equation in the form of `y = mx + c` and made up some random points for `x` and `c`. I calculated the equation and added a small random noise for about 1000 times to generate 1000 data points to feed to my library.

In the made up setup, the library was giving near perfect answer and I was very happy about it.

I checked the inventory and here it was 
1. A `Tensor` library to perform the math
1. A `Gradient Descent` function to converge to the solution

Well enough for the trivial use case. I connected all the dots and started taking the next step.

## Starting with real data
The solution was working on a made up data set. Hence, I decided to give it a spin on real data set. I searched Kaggle to find some data source, checked a few and decided to use one with 414 data records, divided it into train and test set - 321 and 93. Started my test on the library.

Rabbit hole opened up...

### Journey with `NaN`
The first result that I was able to generate with my `prediction` was full of `NaN` values. I was not at all surprised to see that. I have tested it against a trivial made up data set.

Things were bound to break in real world scenario.

I rolled up my sleeves and started debugging process. Put a lot of `println!` statements in every set of lines. The result came faster than expected. At some point, the multiplication was returning very large numbers to the extent, it started producing `NaN`. I checked, my tensors were expecting floating points and they can produce `NaN`.

I started reading about the issue and took some help from CoPilot to understand that the step in the process I was missing is, normalization of data. I put a normization method and invoked it feeding the data to linear regression module.

After this, I was able to see some real numbers instead of `NaN`.

Phew...

### The realization and validation
After fixing the `NaN`, my next validation was to check if the returned results are correct or not. Well, a 70% success rate. Not bad but still felt something amiss. Curious me, opened a python shell this time to conjure a sklearn based linear regression program and fed the same data.

As expected, the python program returned a much less error, around 19% (needless to say the speed difference which I deliberately ignored). I was back in the path of debugging to find out what needs to change for a better accuracy.

### The fix
I look at each line of the code and tried to understand, where am I doing wrong. No error caught my eyes. However, I also noticed that my learning rate was very small - 0.000015 with a very low iteration - 1000.

I bumped up the learning rate, increased it to *0.001*.

Voila, my program producing almost similar result to that of sklearn.

Following is the final result I arrived at,

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




