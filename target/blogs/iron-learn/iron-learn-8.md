# Neural Network in Rust and Proof of Universal Approximation Theorem

Once I was done with the CUDA Integration in Python, this was the time to look back to where I left off with the Rust program. I had to replicate the Neural Network into my Rust Program.

I took a look at the code base, all code was dumped into one single file, violating my personal coding hegeine of 'no more than 500 lines in a file, until absolutely necessary'. Apart from that, the way the logistic regression program was written in raw CUDA launches without following any structure, made me concerned that I have to write duplicate modules for Neural Network to support both CPU and GPU.

I needed a plan to make things unified in`Tensor` level, beyond that, things can differ but any high level implementation should call `Tensor` modules and things should work without worrying about where the math is being performed.

## The Plan
1. Make a `TensorFamily` enum to determine where the logic is gonna run
1. Write a  `TensorBackend` struct which holds all the methods of `Tensor`
1. Implement the `CpuTensor` and `GpuTensor` for `TensorBackend`
1. Finally, rewrite the `Tensor` back which works as the `Factory`

While doing so, stumbled upon some new learning on Rust - the dynamic trait and another rabbit hole...

## The Pivot
The plan looked simple on paper. However, it did not work in the real code. The compiler came back with multiple errors. I tried to use `dyn` trait for `TensorBackend`. I tried to resolve a few. Some I fixed, I understood few new concepts and why Rust was trying to block me to go for a recursive memory allocation pattern and I got stuck. Compiler was very reasonable but I was being completely unreasonable in my plan.

After around two hours of fighting with the compiler, I again had another thought of shutting down the project. I questioned my choices and left the desk for a walk around the block.

There came the solution, I don't need to make the actual `Tensor` unified. No matter what I did, I would still need to make two different execution methods, one for CPU and another for GPU. The user of the library (ironically, that is just me), would make the choice of using GPU or CPU on their work load. They may have installed high end GPUs but for a simple XOR operation Neural Network test, the GPU will actually be slower. I should not make assumptions and must leave the choice to the user.

With this new found reconcilation, I returned to my desk. I devised a new design altogether, where the reasonable set of methods will be exposed for both hardwares. Only difference would be: the CPU-bound tensor can query the memory immediately and return result  while the GPU based tensor need an explicit D2H data copy mechanism. Until the D2H call happens, all data resides on GPU memory.

With this new idea, I abandoned all my plan and started fresh with writing GPU based program separately.

However, this also did not go well, after few more rounds of error I stopped GPU programming completely.

It was really a devastating and crying moment for me. My dream was shatering in front of my eyes. I knew, no way a heavy workload would be completed by my CPU. I need to work on the GPU side. But something in my mind told me quietly, 'don't worry, you will do it, but just not right now'. Somehow, I followed my inner voice and kept aside my thinking brain for few hours. I wrote the Rust CPU-bound neural network, following the Python script.

After around three hours, I was able to finally run my first Rust Neural Network program. Things went pretty well. To be honest, I never expected it to go so smooth. I ran the XOR test in both Python script and Rust program. They showed not exact but very similar results. It was not the exact same result because, the weight initialization followed random sequence without same seed.

Another success, another idea, another play time...

Approved!!!

## The Universal Approximation Theorem
The Universal Approximation Theorem states that, "Given at least one hidden layer in a neural network and enough time neural networks can approximate any continuous function".

Well, now I have a neural network running and it passed the non-linearity test with XOR operation. I can leave the computer switched on overnight to approximate any function. So, why not try it?

I needed a function to be approximated. I became little dramatic here.

I wrote few chits writing `1` to `10`, `+`, `-`, `*`, `/`, `^` and put them in a bowl and picked up 15 times. Please don't judge me. I still have no answer why I did that.
![Math Chits](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-math-chits.png "Math Chits")

I came up with this equation

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



After an arbitary function output, I tried another function, an image. I took an open source black and white image of 474 * 474 pixels. I converted it into a CSV with x, y coordinates as input and the pixel values as 0 and 1, 1 meaning black pixel and 0 means white.


I started writing a python program which can reconstruct a black and white image by looking at the data.

I came across several problems while doing this.

The image was large 474 * 474 pixels. It was taking long time to train.

If I have to run the program, it starts from epoch zero, adding to the pain.

I thought of adding a load and save weights and biases. So that, in the event of a machine crash, I can resume the learning from the last saved checkpoint data.

Then,I ran it for 10000 turns, then again, from there another 30000 runs. It did not go well.

Then I tried resizing the image to 200 * 200 pixels. It started giving some result. So, I changed the learning rate as well as the epochs and tried to find if it can be supported.

On the way I thought, what if I can build row by row of the image. What if I can change the learning rate on the fly.

I did some research on my thoughts and found about Stochastic Gradient Descent, Lerning rate decay. I applied the cos learning decay function and it smoothly started showing correct result.

After running for almost 1.2 million iterations, could see generated output very close to input. The input was a very complex high dimensional function, far beyond simple XOR gate test set or even the logistic regression dataset. This proves that my neural network was exhibiting properties of Universal Approximation Theorem. I can now use this network to do bigger things.

<!-- Image of generated vs original -->

But before that, out of curiosity, I saved the weights and biases and tried to test it against different blank canvas sizes like 512, 50, 1024 etc. In every blank canvas it drew the image.

So, I basically made a very high complexity, inefficient, uber expensive image scaler.

I posted the image on Reddit and another redditor commented about SIREN. I implemented a SIREN in python and no, it did not work. I finally abandoned the plan of writing SIREN after few failures to pursue something else.

After successfully reconstructing approximately 90% of the image, I finally came back to my original tensor program. I spent days on end to make it a generic module which can work for both CPU and GPU.

1. I made a separate `Tensor` trait
2. I made a CPUTensor
3. I made a GPUTensor
4. Implemented all tensor methods as needed.

After a week's effort, I was able to put everything togethere in place and as expected CPU and GPU tensors started working.

Then came the next shock. GPU Tensor was taking 90+ seconds to run the same network, which my python program is taking only 8.

Another challenge. I tried to look for a reason. I took to the debugging sessions, used nsys profiler and found, the major time consuming part of my application is not Matrix Multiplication but it was memory allocation and deallocation.

I tried to question cupy also needs to do the same to achieve its operations, how is it so fast?

Well, the answer was Memory Pool. I found no direct memory pool application in cust library. So, first thought implementing my own but it was a very painful and error prone approach. So, I started searching more and more. Finally I found cust may not provide GPU memory pools but cuda does. Cuda library have memory pool and cust exposes it as sys module.

I tried to incorporate that. Lot of documentation reading. Finally was able to successfully use the memory pool in a main program and I literally could see hunderds of MBs of GPU memory getting allocated within milliseconds. It made me happy. 

But I was surprised when I incorporated the code in my Tensor library. I needed to keep the pool alive during the execution of the program. So, I had to put it inside the appcontext.

And it started failing. Then, I was introduced to Arc<Mutex<>>. However, this did not solve the problem. The actual issue was somewhat more deeply nested in the Cuda implementation.

This is the first time, I actually worked with Raw Pointer in Rust.

After putting all these pieces together, I faced something else. The size mismatch issue. <<<Cry Emoji>>>
I fixed all the size mismatch issue and started thinking, this vversion will definitely work. It worked but time taken to memory allocation was not going down.

After a whole day's worth of effort, I finally found the hidden issue with Ones function needed for Sigmoid. It was making a new instance of tensor, which was pushing all data to Device, making a host to device copy. I wrote a whole new kernel to initiate memory with provided value.

Then I faced another challenge, the `sum` reduction function (`np.sum()`) was both copying data to host from GPU and pushing it back to GPU with new data. I made a change to do the calculation in GPU, instead of CPU.

The time got down some but I found that at that point, majority of the time spent was done in the cuLaunch, indicating performance issue in the kernels itself.

I tried to change the Matrix Multiplication kernel first. Not much change, dropped from ~45 seconds to ~34 seconds.
Next I changed some other kernels. Not much change, so reverted all these and settled down with only the tiled matrix multiplication.

However, the lag was real. So, I tried to use cublas. It seemed to me, cust library does not support cublas. So, I had to resort to cudarc to get CUBLAS working. The road was not smooth. However, after 2 hours of hiccups, I finally ran CUBLAS but I was heart broke, as CUBLAS Version took almost the same time and sometimes more than my naive impl. I tried to profile. Nothing suspicious.

Then I started reading about it. cuBLAS works best with really big matrices, which is not the case here.

After all these revelations, I focused on some refactoring. The whole code base was a mess. I needed some clean workspace to again refocus.

Suddenly it jumped to my mind that float64 might be the bottleneck in my Tensor Library. So, I switched to the python program to confirm my doubt, I switched everything to float64, instead of float32 and it started taking more than 50 seconds, far more than my Rust Tensor program. 

That was it. I switched back to my rust program to fix the remaining clean up, refactoring and build issues.

While separating the network layers, builders and loss functions, I struggled to fix the Generics Issues. I fixed a lot of them.

But final blow came to it when I found that, the neural network I have built heavily relies on floating point mathematics. Especially, gradient descent, learning rate, scaling everything works on floating point mathematics. I had a vague idea on how am I going to solve the problem. I do all my maths in floating point and then round of the final result to Integer.

Then, out of curiosity, I started finding about it, how Integer neural networks work. I came to an idea that, my methodology works but instead of rounding, some other idea quantization is at play. I did not bother to look for it. Will know in the due course of time.

At that point, my main motto was to fix the build issues, generics issues and make my program running again. And probably, use f32 instead of f64.

I made all the changes. It was around 30 seconds, and the tensor program started failing with IllegalMemoryAddress and again return values were NaN once again.

I learnt Rust, fought with the compiler, worked with FFI, invoked external device, wrote CUDA Programs, saw some success. Pretty much a lot of understanding of Rust now.

I agree, Iron Learn was too ambitious as a goal to pursue, that too single handedly. I think, it's better I shut it down at this point. 

I am just taking that. No more Iron Learn.

<!-- Note to self -->
Need to implementation

Sum Axis
Broadcasting. It actually copies to make both m * n and applies the element operation.
All Tests need to be fixed.
Even though we work with Integers, neural network internally converts every element to float.
Need to check if F32 is faster than F64 on GPU and may be that's why the performance issue.
Python code now takes approx 30 seconds for float64 the same program which was taking only 8 seconds in float32 version