# CUDA Integration with Python

After successfully running the Neural Network and testing it with XOR operation and finding a successful result, I became very excited. I had to replicate the same in Rust now but my mind won't let me do that. Whenever, I tried to write a single line of Rust code, my mind wandered 'what would happen, if I feed it a complex equation?'. It started questioning, 'what would happen if I run the linear regression data through neural network?', 'what would it do with the logistic regression dataset?', 'will it work with CUDA?', 'will the regression program work with CUDA?'

So many unanswered questions. It broke my flow.

## The CUDA based Linear Regression Program

As I was already working with CUDA at that, the most low hanging fruits for me were to integrate CUDA in both python and RUST program for everything. I switched over to the linear regression program to execute it on gpu. 

(Un)surprisingly, this did not work. My CPU bound program was giving very low MSE - 7.75 but GPU bound program was returning 40.

Debugger Hat...

After searching the code line by line, I found I was returning a zeroed matrix back from GPU in my linear prediction function. I fixed it to return the actual output instead and it started behaving normal.

```shell
Results:
Total samples: 93
Mean Squared Error: 59.6779
Root MSE: 7.7251
```

Similarly, I wrote another CUDA program for logistic regression and it also went fine. Unfortunately, I somehow missed capturing the results of the CUDA Logistic Program.

One question shot down: Rust CUDA program can work with the regression datasets. Let's move on.

## CUDA integration with python
Once I was happy with the results, I turned to the python neural network script which runs the simple XOR test dataset. I already had worked with bigger datasets before. So, running XOR test dataset, did not feel right to me. I planned to run the linear regression and logistic regression datasets on the neural network too. Ideally, they should run properly without issue.

I wired the JSON file to the script and started running the neural network. 

Oh boy, I just opened another rabbit hole.

No seriously...

On a huge data load, CPU is not powerful enough to work sequentially, even though all the miniscule parallellization numpy does on the array to distribute the load across CPUs. Aparently, the email spam/ham dataset was enough to trigger the dataload, my CPU could not handle anymore. I tried to find a solution in the scikit-learn. Found that, the library has limited GPU Support through `cupy` and there are some setup challenges. As I already understood the maths and I already had a `numpy` version of Neural Network program handy with me, it was easier for me to switch my `numpy` program to `cupy`. Obviously, I would miss a lot of optimizations but that would give me the push to learn optimization techniques down the road too.


Solution was planned, I shot my WSL, installed `cup`y and replaced `numpy` with cupy in the script and there lied another quick sand which drowned me again.

The same program that worked within 10 minutes with thousands of iterations in numpy version was not even running 1000 iterations in 10 minutes in cupy version. Thankfully, been there earlier with my RUST code. It clicked me almost instantaneously about host to device and device to host data copy overhead.

I ran back to the library documentation, found a fix for that and applied it.

## The take away

After fixing the H2D and D2H copy overhead, the program worked decently. I was able to run the same script now within 10-15 seconds, a mere 60x speedup.

Honestly, I got addicted to the speed...

After a long day of fighting through setup and debugging, a little play time was approved.

I just started playing with it. First a single layer. Not much load. I then added two more layers, just for fun.

With this, the neural network brought down the MSE of the linear regression dataset even lower

```
Test MSE: 52.3265 (From earlier 59.6779)
Test MAE: 5.3395
```

The effort was worth it once again. Seeing the network learn step by step in each output, was satifying. And with the speed, I can choose a lower learning and higher epoch (number of training iterations), the network configurations, number of layers, number of nodes in each layer etc.

```
Epoch 1/200000 | Error: 25.926468
Epoch 1000/200000 | Error: 0.482236
Epoch 2000/200000 | Error: 0.414885
Epoch 3000/200000 | Error: 0.377820
Epoch 4000/200000 | Error: 0.354329
Epoch 5000/200000 | Error: 0.340112
Epoch 6000/200000 | Error: 0.331060
Epoch 7000/200000 | Error: 0.324392
Epoch 8000/200000 | Error: 0.319276
Epoch 9000/200000 | Error: 0.315130
Epoch 10000/200000 | Error: 0.311793
Epoch 11000/200000 | Error: 0.308888
Epoch 12000/200000 | Error: 0.306242
Epoch 13000/200000 | Error: 0.303405
Epoch 14000/200000 | Error: 0.300487
Epoch 15000/200000 | Error: 0.298240
Epoch 16000/200000 | Error: 0.296392
```

While looking into the errors, I noticed how Gradient Descent works. At first, the errors are high and and the network corrects quickly towards convergence but as time passes by and the network learns, the errors go down and so does the corrections.

I tried different learning rates. When I chose a bigger learning rate, the neural network was not converging, it was going back and forth between two points and finally returned more errors. On the other hand, having a smaller error rate, gave me a smoother convergence but took very long to converge.

Another revelation was that, for smallest learning rate of 0.005, after around 20000 epochs, the network almost stabilizes but neural network can oscilate with that too.

![Epoch-error plot](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-7-epoch-error.png "Epoch vs Error plot")

Another observation was that running more and more training loops does not change the Mean Absolute Error too much. At some point, saturation is inevitable.

The best result I got with 2 hidden layers are as follows:

```
Training completed in 285.5212 seconds.

Final Results after 100000 epochs and learning rate 0.001:

Test MSE: 46.2653
Test MAE: 5.2730
```

This result is not vastly different from,

```
Training completed in 117.2746 seconds.

Final Results after 40000 epochs and learning rate 0.005:

Test MSE: 52.6716
Test MAE: 5.3199
Starting training...
```

I also tried removing layers, the best result I got using just one hidden layer is as follows:

```
Training completed in 77.6143 seconds.

Final Results after 40000 epochs and learning rate 0.005:

Test MSE: 45.5669
Test MAE: 5.1707
```

By the time, I was done with all my experiments, it was already 6 hours passed. All these insights helped me understand Neural Net a little better than before.

I always doubted the performance of my earlier Logistic Regression program written in Rust. With successful CUDA integration of my program, it was time to do a fair comparison. I made a logistic regression program in the neural network by just using one layer of linear layer and one layer of sigmoid layer.

The result surprised me, my inefficiently written CUDA program was actually sailing faster than my `cupy` program.

I did not bother to find the reason, I was done for the day.

Before I log off for the day, here is a quick inventory check:

1. A CPU Linear regression
2. A CPU Logistic Regression
3. A GPU Linear Regression
4. A GPU Logistic Regression
5. A Python script to run GPU powered neural network

## The Refactoring Phase
After I was done playing with the python script for some time, finally I jumped back into my Rust Program. All the code was written inside one single big 500+ line file, ignoring my personal coding hygeine. Hence I started tidying up things a little.

I also needed to make the gradient descent and all other modules hardware agnostic. They just call tensor methods and things should work for them, without worrying about backend implementation.

The Plan
=========
1. Make a `TensorFamily` enum to determine where the logic is gonna run
1. Write a  `TensorBackend` struct which holds all the methods of `Tensor`
1. Implement the `CpuTensor` and `GpuTensor` for `TensorBackend`
1. Finally, rewrite the `Tensor` back which works as the `Factory`

While doing so, stumbled upon some new learning on Rust - the dynamic trait and another rabbit hole...

The Pivot
=========
1. I was dumped with multiple errors while I tried to use `dyn` trait for `TensorBackend`. I tried to resolve a few. Some I fixed, I understood few new concepts and why Rust was trying to block me to go for a recursive memory allocation pattern.

After spending around 2 hours, a question came to my mind, why even I am trying to that. The way CPU and GPU calculations are going to happen are completely different.

I can send back the result from my CPU based program after each calculation but GPU based program should not return result after every operation, rather it should return result only when asked for.

With this new idea, I abandoned all my work and started fresh with writing GPU based program separately. If I need to unify the logic at some point, I will do it then. Now is not the time.

After few more rounds of error, I stopped completely. Then after a short walk around the block, this came to my mind, how about I run the neural network on CPU first and then integrate to GPU later.

With that thought, I wrote the rust cpu based program following the python neural network script.

I had to fight with the compiler for quite a few issues and then, I had to write a few new methods too. I got to know about the difference of numpy in syntax of - @ and *.

After around 3 hours, I was able to finally run my first Rust Neural Network program and was showing similar results like the one in python program.

This thought came to my mind. Neural networks are good in estimating functions. What if I write any arbitary random function and sample points from the graph and try out the test again.

I came up with this equation (don't try to put your mind in it, its a garbage function)

```
a(x)=((x^(2)+x^(3)+x^(4)+x^(6))/(x^(5)))
```
And this gave me this plot

![alt text](actual-plot.png)

I made an arbitary rule, the blue points are true and red points are false.

Then I sampled some 25 points for training data and 6 for testing. First attempt went unsuccessful. I could not find the reason. I tried it with my python cupy program. That also failed. I then wrote a sklearn script and it went successful. 

This came to me as another challenge. After doing some tweaks, I found the issue, I was normalizing the input and denormalizing the output, which made the prediction result incorrect. I removed the normalization and denormalization layer and it started working.

A fair result for 10000 epoch and 30 data points

```shell

╔════════════════════════════════╗
║ Iron Learn v5
║ Mode: CPU
║ Learning Rate: 0.0001
║ Epochs: 100
║ Data Path: data.json
╚════════════════════════════════╝

Predicted: 0.0000, Actual: 0.0000, ✓
Predicted: 0.9999, Actual: 1.0000, ✓
Predicted: 1.0000, Actual: 1.0000, ✓
Predicted: 0.0000, Actual: 0.0000, ✓
Predicted: 0.0024, Actual: 0.0000, ✓
Predicted: 0.0000, Actual: 0.0000, ✓

```

