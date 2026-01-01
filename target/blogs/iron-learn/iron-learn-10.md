## The Refactoring

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

