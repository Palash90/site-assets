# Building the First Neural Network
My program was running succesfully and was using GPU to do the heavy lifting and bring the complexity down from $O(n^3)$ to $(n)$. I was happy, I integrated the logistic regression and used the GPU there as well. I know my GPU kernel programs are still not the best in optimization standards, but I really don't care too much at this point. I had sufficient performance boost to run 20000 loops of training runs under 10 seconds, which was taking 1 hour few days ago.

I will rather deep dive in learning the next step of machine learning than making my code super efficient.

But there is no harm in doing some kind of baseline check while my machine does the hard work and I take rest. It was so satisfying to see GPU getting used by my own program, I could not resist the urge, I bumped up the iteration loop to 10 million epochs. My GPU frequently went 100% usage. Wow!!! A cherishable moment.

10 million iterations took almost 45 minutes in my machine, still less than the time it took for CPU to run only 5000 iterations. Accuracy also hit 93.09% this time.

After playing for some other data and few hyperparameter tuning, I took my next step in the journey. To build a neural network.

At this point my inventory include quite a few things actually.

1. An almost accurate CPU Powered Tensor library
2. A Gradient Descent implementation which runs on CPU
3. Few reusable cuda programs
4. A fully fledged GPU powered Tensor kernels. 
5. A complete orchestrator of Gradient Descent
6. A full logistic regression program which returns accuracy almost identical to other libraries.

I started with a python script and eventually rewrote it in rust.

**Some mathematical idea of what does a neural network do**

Basically what it does is as follows ====>

It takes the linear algebra layer (matrix multiplication) and wraps a non-linear function around it.

**This is the moment you should describe why we need non-linearity**

**This is the exact moment, you should bridge the gap with multi-variable calculus and chain rule**

I have fed it the XOR dataset to test. I reran the program and it ran successfully.

It sparked my curiosity, what happens if I feed it the real email dataset. As I thought, so I did. Well, it took me a while to get that working with real dataset of pass fail and email spam but it worked on both and I got 92.85% accuracy on 1000 training iterations and 0.1 learning rate.

I thought of running it against the linear data set and see what happens. It failed when my activation function was sigmoid. I had to make a few changes in the program to support linear output.

However, I noticed, without GPU support, even my simple numpy based neural network script also crashed my machine. I know numpy under the hood does many CPU optimizations but they are peanuts in front of massive calculation load of O(n^3).

The initial success of the python program pushed me to start a new journey.