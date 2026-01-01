# Generating Simba Network with Rust
After successfully approximating the lottery math function (check previous post for context), I decided to challenge the network more. There are so many functions to choose from. Mathematics has evolved a lot from the inception and honestly I know a very little about them. Proving them would be another chore for me. I needed something interesting, something exciting. It struck me, image is nothing but a 2D plot of an arbitary and arguably very complex function. 

What if I can ask the network to approximate it?

I found a Lion Cub drawing in Black and White and used the following encoder/decoder script to encode/decode pixels into numbers and vice versa: [Image to CSV Encoder Decoder](https://github.com/Palash90/iron_learn/blob/blog-8/python_scripts/image_inputs/image_to_csv.py)

Once done, I again launched my Python script to feed the generated CSV data to the neural network.

## Helping Machine to Draw
The script struggled at lot of points and I had to fix those to help machine learn how to draw Simba.

### The Large Image Issue
The original downloaded image was a large 474×474 pixels. It was taking very long to train. To avoid this issue, I had to resize it to 200×200 pixels.

### The Machine Crash and Restart
I had occassional machine crashes due to overheating and every time that happened training starts from scratch. This was a huge waste of time and resources. So, I added a save/load mechanism to resume the learning from the last saved checkpoint. The machine can now take a pre-saved model and can start from there.

### The Error Oscillation
No matter how small learning rate I chose, the training always was getting stuck into the error oscillation loop. At one point, it occured to me that, if I choose very small learning rate like 0.000001 I could save the training but definitely, that would take me longer. Then I thought, what if I can gradually decrease the learning rate programatically. I did some research on my thoughts and found about Cosine Annealing. I applied the cos learning decay function and it started showing smooth learning.

```python
decay_factor = 0.5 * (1 + math.cos(math.pi * i / (epochs+epoch_offset)))
current_lr = lr_min + (lr_max - lr_min) * decay_factor 
```

## The Result: Art through Math
After fixing all the issues and running the network for almost 1.2 million iterations (around 4 hours on my machine), I could see generated output very close to the input image. The input was a very complex high dimensional function, far beyond simple XOR gate test set or even the logistic regression dataset. This proves that my neural network was exhibiting properties of Universal Approximation Theorem. I can now use this network to do bigger things.

For comparison, here are the results:
### Original Image
![Original Image](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-original.png$ "Original Image")

### The Initial Static
![Initial Static](${iron-learn-9-plot_statics}$)

### Final Image on 200x200
![Final Reconstruction](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-plot_final.png$)

### Reconstruction on Higher Resolutions
At that point, I was pretty sure, the network learnt the underlying function. With that confidence and the saved weights, I tried to test it against different blank canvas sizes like 512, 50, 1024 etc. In every blank canvas it drew the image. 

Following is the same image reconstructed on 1024x1024 resolution:

![Higher Resolution Reconstruction](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-plot_FINAL_1024.png$)

Since the network had learned the mathematical concept of the lines rather than just the pixels, the 1024x1024 version didn't look 'pixelated' like a standard zoom—it looked like the network was redrawing its own masterpiece on a bigger canvas.

## Validation and Next Steps
I basically made a highly complex, inefficient, uber expensive image scaler. The result was satisfying but not perfect. It proved the point but I needed perfection.

I posted the resultant image on Reddit and another redditor commented about `SIREN` or SInusoidal REpresentation Networks.

`SIREN` is a neural network which uses Sine activation function instead of ReLU or Tanh. Mostly used for purpose of Implicit Neural Representation, a technique very similar to what I was trying to achieve. `SIREN` is more effective than other neural networks in representing Images, Audios etc.

I implemented a `SIREN` in python with the hope of reconstructing the image to a more perfect one. But my efforts were in vain, it did not work. I finally abandoned the plan of writing `SIREN` after few failures to pursue something else.

While my network used Sigmoid, SIREN uses Sine waves, which are naturally better at capturing the 'sharp edges' of a drawing. Even though I failed to implement it then, it changed how I looked at the 'frequency' of my data.

In my next attempt, I actually achieved a sharper, more detailed reconstruction.

## The Rust Comeback
The idea I implemented showed some fruiful results. The success of the Python script rejuvenated my thoughts. I was ready to take next challenge. I was again ready to pour some energy into the Rust program. A small teaser, I would have missed the adventure and the learning if I did not come back to Rust.

The journey again resumed which were paused for few days. I wore the Rustacean hat and after around three weeks, I had this drawn by my machine:

![Final Drawing by Rust](https://github.com/Palash90/iron_learn/blob/main/image/images/output800000.png)




























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

