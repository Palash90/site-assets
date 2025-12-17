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