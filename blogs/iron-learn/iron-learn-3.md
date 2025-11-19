My matrix multiplication method was taking forever to complete even for a small matrix like 321 * 6. I had to wait for long time to see my work to completion.

I have a CUDA GPU, I was just feeling lazy to configure it for programming for 2 years. However, I overcame my laziness. I started looking at places how to start using GPU in rust code. Here is how it went.

1. I started looking around internet.
2. I found many references to ndarray.
3. I was not sure, if ndarray will still use CPU or it will utilize gpu
4. So, I searched more and stuck with rusta-cuda
5. Well, at this point things took a down turn.
6. I spent 3 to 4 hours just to see my GPU actually doing something.
	a. I installed CUDA Toolkit
	b. I installed cmake
	c. I installed MSVC 19
	d. I installed Visual C++ build tools
	e. Finally I was able to run the cuBLAS matrix multiplication program
7. Man the effort was worth it.
8. A 1280 * 960 vs 960 * 640 matrix multiplication just took 0.619 msec and it did not even break a sweat
9. Curious me tried to hit some more numbers - 12800 * 9600 vs 9600 * 6400 matrix multiplication
10. I was amazed by the result. It took only 392.912 msec.

For the first time in 2 years, I am using my nvidia GPU the exact way I initially thought I will use it for.

Then the next step started. How to write a program that runs on GPU.

1. I started with rust-cuda
2. After spending hours on it and no success, I resorted to other libraries
3. A quick google search gave me few options
4. I tried cust, rust-gpu and some random internet suggestions.
5. cust worked
6. I started with simple vector addition
7. Then switched to matrix multiplication
8. Started small with 2 * 1 vs 1 * 2 with small numbers
9. Ran against random numbers
10. Now running against big set - 1024 * 1024 vs 1024 * 1024 - took few milliseconds
11. Finally, I could do it.
12. Next step - integrate this into my repo


