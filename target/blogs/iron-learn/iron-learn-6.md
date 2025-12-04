# Building the First Neural Network
My program was running successfully and was using GPU to do the heavy lifting and bring the complexity down from $O(n^3)$ to a highly parallelized, faster execution. Having successfully integrated the CUDA code into the logistic regression module, I was happy. I know my GPU kernel programs are not the best in optimization standards, but I really didn't care too much at that point. I had sufficient performance boost to run 20000 training loops under 10 seconds, which was taking almost 4 hours a few days ago.

I'd rather deep dive into learning the next step of machine learning than focus on making my code super efficient.

But there is no harm in doing some kind of baseline check while my machine does the hard work and I take rest. It was so satisfying to see GPU getting used by my own program, I could not resist the urge, I bumped up the number of epochs to 10 million. My GPU frequently went 100% usage. Wow!!! A cherishable moment.

10 million iterations took almost 45 minutes in my machine, still less than the time it took for CPU to run only 5000 iterations. Accuracy also hit 93.09% this time.

After experimenting with other data and fine-tuning a few hyperparameters, I took my next step in the journey - building a neural network.

At that point my inventory included quite a few modules:

1. An almost accurate CPU Powered Tensor library
1. A Gradient Descent implementation
1. Few reusable CUDA kernel programs
1. A complete Gradient Descent orchestrator
1. A logistic regression program which returns accuracy almost identical to `sklearn` library.

## The Maths Behind
The main objective of machine learning is to minimize the loss, calculated by the difference of actual test values and predicted values. To do this we heavily rely on mathematics.

### Linear Regression
For a refresher, let's revisit how Linear Regression works.
The objective of Linear Regression is to find a fitting straight line through the data. The equation for a line is represented as follows:

$$
y = mx + c
$$

In linear regression, we have to find the $m$ and the $c$ from $y$ and $x$ where $x$ is input data and $y$ is the output for that $x$.

Linear Regression starts with the inputs($x$) and some random weight matrix ($m$) and bias matrix ($c$) and follow a series of matrix multiplication and addition routine to arrive at the prediction. Then it checks the predicted result with actual target values, taken from the test data and calculates the loss.

![Linear Regression](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-6-linear_regression.png "Linear Regression")

The loss is then used to manipulate the weights using calculus by finding the minima. And it does the same process over multiple times. Eventually the predicted output starts matching very close to the actual output.

### Logistic Regression
The logistic regression also does the same thing, only with an extra step. It does the same $y = mx + c$ calculation and then it wraps the result into a sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

![Logistic Regression](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-6-linear_regression.png "Logistic Regression")

Then it measures the loss and follows the same process as earlier to get into an optimal solution.

### Neural Network
So far so good, we can predict a continuous variable which follows a straight line, we can predict a binary value.

However, in real world, not everything is either a line or in simple $true$/$false$ category. We can have 

**This is the moment you should describe why we need non-linearity**

**This is the exact moment, you should bridge the gap with multi-variable calculus and chain rule**

I have fed it the XOR dataset to test. I reran the program and it ran successfully.

It sparked my curiosity, what happens if I feed it the real email dataset. As I thought, so I did. Well, it took me a while to get that working with real dataset of pass fail and email spam but it worked on both and I got 92.85% accuracy on 1000 training iterations and 0.1 learning rate.

I thought of running it against the linear data set and see what happens. It failed when my activation function was sigmoid. I had to make a few changes in the program to support linear output.

However, I noticed, without GPU support, even my simple numpy based neural network script also crashed my machine. I know numpy under the hood does many CPU optimizations but they are peanuts in front of massive calculation load of O(n^3).

The initial success of the python program pushed me to start a new journey.