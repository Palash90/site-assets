I again started my work on the library. It was silently killing me from inside that, I have a GPU that I bought to do these calculations for me. Just for my laziness I am wasting my time in waiting rather than learning.

Seriously, the wait time to see anything descent was beyond my nerves and also I had to always go with smaller dataset, as my machine could not cope up with that much of data.

I rolled up my sleeve again to find a solution. If nvidia CUDA examples can run on my machine, why can't I take the step forward to program on my own?


I noticed, with my GPU and nvcc 12.0+, I can access something called as cuBLASDx, a GEMM (GEneral Matrix Multiplication) wrapper. I thought of giving it a try.

I started reading the documentation and guide hosted on NVIDIA website.

First thing to get hold of is the library itself. It does not come default with CUDA Toolkit. I downloaded and the second thing I was missing was GCC 7+. I had GCC 6.3.0 and MSVC but MSVC was not supported by the cuBLASDx. So, I had to download GCC 19

Next was to install MathDX Package, which I had to download too.

After downloading all the required softwares, finally when I tried to compile my CUDA program, I got into lot of issues around C++ installations. It was a real frustrating experience. I installed different versions of Visual Studio. I could not understand what happened. Finally left the thought of writing the program.

Then came to my mind, Linux is usually easier in these aspects. Hence, I started the store app, installed Ubuntu on my Windows and installed all required modules and it worked. No missing package error, no missing software error.

However, the error, I got broke me a little. My GPU was built with Ampere Architecture but cuBLASDx works with sm_70 or higher architecture.

Hence, I can't run cuBLASx GEMM modules on my GPU.

