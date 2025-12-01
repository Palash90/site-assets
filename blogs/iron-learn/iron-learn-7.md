It's time to write the neural network module in Rust, following the similar logic in the python script.

But something caught my attention. Before proceeding with neural network I wanted to check, what happens when I send linear regression to GPU.

To verify, I first installed cupy for python so that, my python scripts can also use the power of GPU

I switched over to rust program to implemnt the linear regression on gpu

Surprisingly, this did not work. My CPU bound program was giving very low MSE - 7.75 but GPU bound program was going to 40

Another debugging time.

After searching for line by line, I found that, I was returning an zeroed matrix back from GPU in my linear prediction function. I changed to the actual output and then it started behaving normal.

```shell
Results:
Total samples: 93
Mean Squared Error: 59.6779
Root MSE: 7.7251
```

Once I was happy with the results, I turned to the python neural network script. I found that, it only runs a simple XOR test dataset. I already had worked with bigger datasets before. So, running XOR test dataset, did not feel right to me. I tried to run the linear regression and logistic regression datasets on the neural network. Ideally, they should run properly without issue.

I wired the JSON file to the script and started running the neural network. Oh boy, I just opened another rabbit hole.

No seriously man...

After an hour of internet research, I found that cupy does the trick. I shot my WSL, installed cupy and replaced numpy with cupy in the script and there it was.

The same that worked within 10 minutes in numpy version was not even running 1000 iterations within 10 minutes. Thankfully, been there earlier with my RUST code. It clicked to me almost instantaneously about host to device and device to host data copy overhead.

I ran back to the library documentation and found a fix for that and applied it.

This started back decent. I was able to run the same script now within 10-15 seconds.

Honestly, I am getting addicted to the speed...

I just started with single layer. Not much load. I then added two more layers, just for fun. The linear regression output brought down the MSE even lower

```
Test MSE: 52.3265
Test MAE: 5.3395
```
The effort was worth it once again. Seeing the network learn step by step in each output, was literally satifying. And with the speed, I can choose a lower learning and higher epoch (number of training iterations), the network configurations, number of layers, number of nodes in each layer etc. I just took a pause to this satisfying phenomenon by playing with the neural network.

Don't worry, all these concepts will be made clear very soon.

I got some output like following, which got me stuck for some time before I moved on to the Rust program.

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

While looking into the errors, I also noticed how Gradient Descent works. At first, the errors are high and and the network corrects quickly towards convergence but as time goes by and the network learns, the errors go down and so does the corrections.

I also tried with different learning rates. When I chose a bigger learning rate, the neural network was not converging, it was going back and forth between two points and finally returned more errors.

While, having a smaller error rate, gave me a smoother convergence but took very long.

Another revelation was that, for smallest learning rate of 0.005 after around 20000 epochs, the network almost stabilizes.

With that low learning error also, neural network can oscilate.

[Image = iron-learn-7-epoch-error.png]

And also observed, running more and more training loops does not change the Mean Absolute Error too much.

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

By the time, I was done with all my experiments, it was already 6 hours passed. But I was very happy. 

For two reasons -
1. I actually ran a whole neural network and saw it learn from data and its mistake
1. I was thinking, my inefficiently written CUDA program is taking higher time. But that actually was not the case. I tried with python and cupy, that took more time to complete.

**I was surprised. My program was taking less time than cupy program**

Anyways, I was done for the day.

At this point the inventory looked like this.

1. A CPU Linear regression
2. A CPU Logistic Regression
3. A GPU Linear Regression
4. A GPU Logistic Regression
5. A Python script to run GPU powered neural network


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