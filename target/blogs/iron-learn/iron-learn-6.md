# Building the First Neural Network
My program was running successfully, using the GPU to do the heavy lifting and reducing the time complexity for the core matrix multiplication operation from $O(n^3)$ to a highly parallelized, faster execution time. Having successfully integrated the CUDA code into the logistic regression module, I was happy. I know my GPU kernel programs aren't the best in optimization standards, but I really didn't prioritize that at that time. I had sufficient performance boost to run 20,000 training loops under 10 seconds, which took almost 4 hours a few days ago.

I'd rather deep dive into learning the next step of machine learning than focus on making my code super efficient.

But there is no harm in doing some kind of baseline check while my machine did the hard work and I took rest. It was so satisfying to see GPU getting used by my own program. I could not resist the urge, so I bumped up the number of epochs to 10 million. My GPU frequently hit 100% usage. Wow!!! A memorable moment.

10 million iterations took almost 45 minutes in my machine, still less than the time it took for CPU to run only 5000 iterations. Accuracy also hit 93.09% this time.

After experimenting with other data and fine-tuning a few hyperparameters, I took my next step in the journey: building a neural network.

At that point, my inventory included quite a few modules:

1. An initial accurate, CPU-based Tensor library.
1. A Gradient Descent implementation
1. A few reusable CUDA kernel programs
1. A complete Gradient Descent orchestrator
1. A logistic regression program which returns accuracy almost identical to `sklearn` library.

With the performance challenges temporarily solved, it was time to move on from logistic regression and truly begin my journey into neural networks. To do that, I first needed to solidify my understanding of the fundamental mathematics.

## The Mathematics Behind
The main objective of machine learning is to minimize the loss, calculated by the difference between actual target values and predicted values. To achieve this, we heavily rely on mathematics.

### Linear Regression
For a refresher, let's revisit how Linear Regression works.
The objective of Linear Regression is to find a fitting straight line through the data. The equation for a line is represented as follows:

$$
y = mx + c
\text{Or as better known to programmers: }
y = W^T x + b
$$

In linear regression, we have to find the weights $W$ and the bias $b$ from $y$ and $x$ where $x$ is input data and $y$ is the output for that $x$.

Linear Regression starts with the inputs($x$) and some random weight matrix ($W$) and bias matrix ($b$) and follows a series of matrix multiplication and addition routine to arrive at the prediction. Then it checks the predicted result with actual target values, taken from the training data and calculates the loss.

![Linear Regression](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-6-linear_regression.png "Linear Regression")

The loss is then used to update the weights using calculus by finding the minimum (the gradient). This process is repeated multiple times. Eventually, the predicted output starts matching very close to the actual output.

### Logistic Regression
The logistic regression performs the same initial calculation $y = mx + c$, but extends the process with an extra step: it wraps the linear output into a sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

And here is how the logistic regression process looks like:

![Logistic Regression](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-6-logistic-regression.png "Logistic Regression")

Then it measures the loss and iteratively follows the same process to find an optimal solution.

### Neural Network
So far, we can predict a continuous variable (Linear Regression) and a binary value (Logistic Regression).

However, in real world, not everything is either a line or in simple $true$/$false$ category. We can have data that follows a non-linear path. So, we need to introduce non-linearity in our process too. We do this by wrapping the linear function output in a non-linear activation function. It has been observed that by stacking multiple layers of chained linear and non-linear activation functions, we can approximate any arbitrary function by looking at the input and output, a.k.a. **Universal Approximation Theorem**.

That's exactly what is done in a neural network.
![Neural Network](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-6-nn.png "Neural Network")

Well, so we get the non-linear output following a series of different linear and non-linear function outputs. But, to minimize the loss, we have to update the starting weight and bias matrices. But the loss will only be calculated in the end, after the prediction is made.

What could be the solution?

Seems like centuries ago great mathematicians have already solved the problem. The answer is the Chain Rule:
$$
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

**Example**
$$
\begin{aligned}
\text{Let } y &= (3x^2 + 5)^4 
\text{So, } f'(g(x)) = 4 \cdot (3x^2 + 5)^3 \text{and} g'(x) = 6 \cdot x
\text{then} \quad \frac{dy}{dx} &= 4(3x^2 + 5)^3 \cdot (6x) \\
\text{or} \quad \frac{dy}{dx} &= 24x(3x^2 + 5)^3
\end{aligned}
$$

This formula shows us how the change in early layers impacts the final output. Following this logic, we calculate the loss at the final step. We then use the Chain Rule to propagate this loss backward through the layers to update the weights in the first layer, thereby minimizing the loss.

That's why we call this step of the calculation the Back Propagation algorithm.


## New Journey Begins

Once I understood the math behind, I started implementing it. At first, I wrote my idea in a Python script to quickly prototype the solution.

To my surprise, it started working in the first attempt with the XOR dataset. I could not believe my own eyes.

It sparked my curiosity: what happens if I feed it the real email dataset? As I thought, so I did. Well, it took me a while to get that working with the real dataset of pass-fail and email spam but it worked on both and I got 92.85% accuracy on 1000 training iterations and 0.1 learning rate.

I then decided to run it against the linear dataset. As expected, it failed when run with the sigmoid activation function, as sigmoid forces the output between 0 and 1, which is incorrect for continuous (linear) data.

However, I noticed that without GPU support, even my simple NumPy-based neural network script also crashed my machine. I know $\text{NumPy}$ performs many CPU optimizations under the hood, but they are negligible compared to the massive computational load of $O(n^3)$. This makes GPU acceleration mandatory to move ahead with this project.

The initial success of the python program pushed me to start a new journey...
