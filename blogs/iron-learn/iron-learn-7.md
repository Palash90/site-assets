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

![Epoch-error plot](${iron-learn-7-epoch-err} "Epoch vs Error plot")

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

