# Integrating GPU Matrix Multiplication
At this point, the inventory looks like the following
1. A rudimentary Tensor Library
2. Gradient Descent implementation without any optimization
3. A separate project which has a running GPU Matrix Multiplication program

As my CPU based program taking longer, I thought of changing the Matrix Multiplication routine to utilize GPU. The moment, I put the cust code in my program, it started giving me angry red eyes and screamed at me with multiple errors. The compiler correctly pointed out that `DeviceCopy` trait from cust library has not been implemented for my type.

Ah, the classic trait bound error which I almost forgot after working in python and JS for last 14 months. Rust is so secure, it won't let me play with memory carelessly. Well, the `cust` library takes a step forward and makes this even harder for any types which refers to raw pointers. `Vec<u32>` and `Vec<T>` are obviously one of those and these are the backbone of my `Tensor`.

I tried looking around the compiler errors. I started writing the `DeviceCopy` impl blocks for `Tensor`. I went through many error cycles and finally discarded all my changes.

Then it struck me, my `Tensor` type can't implement `DeviceCopy` because it has a reference type - `Vec` inside it. However, I used `Vec` to implement the matrix multiplication on GPU and that also used `Vec`. What was the difference. Curious me again looked around the GPU Matrix Multiplication Code and I found that, it takes a slice of the `Vec<T>`.

```rust
// Allocate device buffers and copy inputs
let mut d_a = DeviceBuffer::from_slice(&host_a)?;
let mut d_b = DeviceBuffer::from_slice(&host_b)?;
let mut d_c = DeviceBuffer::from_slice(&host_c)?;
```

There it was. If my type `T` is a trait that implements `Copy` and is non-refernce by it's type, then I can use the same code.

With this new found learning, I went back to my `Numeric` trait and found that, it is already bound by `Copy` trait and all the known implementations of this trait are already the primitive types which are non-referential already. I added the following trait bound to my `Numeric` trait as follows -

```rust
pub trait Numeric:
    Copy
    + DeviceCopy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
```

And it threw me new error on `Complex` type, which is an implementation of `Numeric` type. I could implement this simply for my `Complex` struct implementation. As the members in the struct are both `f64`, thus already natively implemented in the original `cust` library.

I simply added the `DeviceCopy` to my original list of `derive` implementations.

```rust
#[derive(Debug, PartialEq, Copy, Clone, DeviceCopy)]
```

Voila, no error.

So, now all the trait bounds are implemented correctly. I just need to copy the kernel files and run it.

I wrote a separate GPU Matrix Multiplication function for this - 

```rust
fn _gpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        let rows = self.shape[0] as usize;
        let cols = rhs.shape[1] as usize;
        let common_dim = self.shape[1] as usize;

        let mut data = vec![T::zero(); rows * cols];

        // println!("Launching GPU Kernel for Matrix Multiplication...");
        // Keep the context alive for the whole function scope.
        let _ctx = cust::quick_init();

        let d_a = DeviceBuffer::from_slice(&self.data).unwrap();
        let d_b = DeviceBuffer::from_slice(&rhs.data).unwrap();
        let d_c = DeviceBuffer::from_slice(&data).unwrap();

        let _ctx = cust::quick_init();

        // PTX produced from kernels/matrix_mul.cu
        let ptx = include_str!("../kernels/matrix_mul.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
        let function = module.get_function("matrix_mul").unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        // Kernel launch params (must match TILE used in the .cu)
        const TILE: u32 = 16;
        let block = (TILE, TILE, 1);
        let grid_x = ((cols as u32) + TILE - 1) / TILE;
        let grid_y = ((rows as u32) + TILE - 1) / TILE;
        let grid = (grid_x, grid_y, 1);

        unsafe {
            let result = launch!(
                function<<<grid, block, 0, stream>>>(
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_c.as_device_ptr(),
                    rows as i32,
                    cols as i32,
                    common_dim as i32
                )
            );

            match result {
                Ok(_) => {}
                Err(e) => return Err(format!("CUDA Kernel Launch Error: {}", e)),
            }
        }

        let result = stream.synchronize();

        match result {
            Ok(_) => {}
            Err(e) => return Err(format!("CUDA Stream Synchronization Error: {}", e)),
        }

        let result = d_c.copy_to(&mut data);

        match result {
            Ok(_) => {}
            Err(e) => return Err(format!("CUDA Device to Host Copy Error: {}", e)),
        }

        Ok(Tensor {
            shape: vec![rows as u32, cols as u32],
            data,
        })
    }

```

Ran the program and to my surprise, it took longer than the CPU Multiplication program and it seemed like my GPU is blocked for quite some time. Something amiss for sure. I halted the program after waiting around 2 minutes.

I looked through the program, I am initializing the context twice in the GPU Multiplication function. I removed one instance and ran it again. Same issue again. Then I thought, isn't it that, my `Gradient Descent` runs in a loop and the multiplication function is called within that loop multiple times. So, the context is initializing multiple times. How about I move the initialization part somewhere else?

I moved the initialization in the entry point itself - the `main` function and ran it again.

The result shattered me - It took 55 seconds, longer than CPU Multiplication and I am back with `NaN` output in linear regression and 30% accuracy for logistics regression.

Debug time...

First I checked, if my CPU Multiplication method still works or not. My trustworthy CPU Matrix Multiplication function still works and gives the same result as before.

Definitely it is the GPU matrix multiplication program that takes longer and still computes inaccurate result.

The first change I made was to bring down the iterations to 10 and print each step in the GPU 

Each multiplication was taking

- GPU = ~750-800 µs
- CPU = ~20 - 25 µs

I dove deeper into the results, added more logs. In conclusion, the most time was taken by data copy between main memory to GPU and GPU to main memory, a fraction of the time was actually taken by the actual matrix multiplication.

Ok, first problem understood. If we can run the whole training loop inside CUDA, we will see performance boost. New plan of action -

1. Copy both the input matrices from json to main memory
2. Copy the same into GPU
3. Run the training loop
4. Get the data back from GPU
5. Get the weight and bias matrix back from GPU
6. Store the results
6. Next time onwards use it for prediction (either through GPU or CPU)

But what about the accuracy part?
Let's dive a little deeper into that.

I ran all the test cases with gpu matrix multiplication. Almost all tests that were associated with matrix multiplication failed. I then switched back to cpu matrix multiplication. It all ran fine.

## The acceptance
I already had spent more than two days around fixing things here and there, integrating CUDA code etc. Now I see, two major challenges, if I have to gain the speed boost, I need to move the whole linear regression code inside CUDA, otherwise time spent in transporting data back and forth CPU/GPU would eat up a lot of time. To do that, I have to write everything in C, which defies my initial purpose of learning Rust and Machine Learning as a whole.

The second challenge is that, I have been very rusty with C language as I have not touched it almost 16 years now. So, I have to learn everything in C and then only I would be able to write CUDA code and then I will be able to continue my learning journey with Rust and ML.

Hence, I accepted that, I won't work on GPU at this stage. Down the line somewhere if I really feel the need for it, I will do it. But till then, I am happy with running small data sets and longer execution time.

