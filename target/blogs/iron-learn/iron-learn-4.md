# The Bubble Burst
The GPU Matrix Multiplication program was a success; or so I thought. At that point, the inventory looked like the following:
1. A rudimentary **Tensor** Library written in Rust
2. **Gradient Descent** implementation without advanced optimizers (like **Adam** or **RMSprop**) - basic but works
3. A separate project with code to run Matrix Multiplication on GPU
4. All the wiring was done using Rust, the `cust` library and the external GPU kernel program

In a nutshell, I was ready to put the GPU performance tweak into my library and roll with speed.

**Copy, Paste, and...**

## Rust Compiler's Full Blow
The moment I put the `cust` code in my library, the Rust compiler started screaming at me with multiple errors. The compiler correctly pointed out that `DeviceCopy` trait from `cust` library has not been implemented for my custom types.

Ah, the classic trait bound error which I almost forgot after working in **Python** and **JS** for the last 14 months. Rust is so secure, it won't let me play with memory carelessly. Well, the `cust` library takes a step further and makes this even harder for any types which refer to raw pointers. `Vec<u32>` and `Vec<T>` are obviously one of those and these are the backbone of my `Tensor`.

I tried looking around the compiler errors. I started writing the `DeviceCopy` impl blocks for `Tensor`. I went through many error cycles and finally discarded all my changes.

Then it struck me, my `Tensor` type can't implement `DeviceCopy` because it has a reference type - `Vec` inside it. However, I used `Vec` to implement the matrix multiplication on GPU and that also used `Vec`. What was the difference? A curious me again looked around the GPU Matrix Multiplication Code and I found that, it takes a slice of the `Vec<T>`.

```rust
// Allocate device buffers and copy inputs
let mut d_a = DeviceBuffer::from_slice(&host_a)?;
let mut d_b = DeviceBuffer::from_slice(&host_b)?;
let mut d_c = DeviceBuffer::from_slice(&host_c)?;
```

There it was. If my type `T` is a trait that implements `Copy` and is non-reference by its type, then I can use the same code.

With this newfound learning, I went back to my `Numeric` trait and found that, it is already bound by `Copy` trait and all the known implementations of this trait are the primitive types which are non-referential already. I added the `DeviceCopy` trait bound to my `Numeric` trait as follows -

```rust
pub trait Numeric:
    Copy
    + DeviceCopy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
```

And it threw me new error on `Complex` type, which is an implementation of `Numeric` type but does not implement `DeviceCopy`. I could implement this simply for my `Complex` struct implementation. As the members in the struct are both `f64`, thus already natively implemented in the original `cust` library.

I simply added the `DeviceCopy` to my original list of `derive` implementations of `Complex` type.

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

I ran the program, and to my surprise, it took longer than the original CPU-based Matrix Multiplication program. It seemed like my GPU was blocked for quite some time. Something was amiss, for sure. I halted the program after waiting for approximately two minutes. It definitely was not the "Flash" performance I expected.

The bubble just burst...

Time for another debugging session.

## The Debugging Phase
Looking through the program, I noticed I was initializing the context twice in the GPU Multiplication function. I removed one instance and ran it again. Same issue again. Then I thought, 'Isn't it that my `Gradient Descent` runs in a loop and the multiplication function is called within that loop multiple times? This had to be the reasonable explanation: the context was being initialized multiple times, causing the delay. How about I move the initialization part somewhere else?'

I moved the initialization in the entry point itself - the `main` function and ran it again.

The result shattered me: It took 55 seconds - longer than CPU Multiplication, and I was back with `NaN` output in linear regression and 30% accuracy for logistic regression.

Debug time...

First I checked, if my CPU Multiplication method still works or not. My trustworthy CPU-based Matrix Multiplication function still works and gives the same result as before.

Definitely it is the GPU matrix multiplication program that takes longer and still computes inaccurate result.

The first change I made was to bring down the iterations to 10 and print each step in the GPU-based function

For each multiplication, the hardware was completing calculations as follows:

- GPU = ~750-800 µs
- CPU = ~20 - 25 µs

Yeah, you read it right. CPU-based program rocks, GPU shocks...

I dove deeper into the results, adding more logs after each step. The logs unmasked the culprit, most of the time was taken by data copy (Host to Device and Device to Host), while only a fraction was actually spent on GPU kernel execution.

Okay, the first issue was nailed down: **data transfer overhead**. I chalked up a plan. If I run the whole training loop inside CUDA, I will see performance boost. 

### A new course of action
1. Copy both the input matrices from json to main memory
2. Copy the same into GPU
3. Run the training loop
4. Get the computed weight and bias matrix back from GPU
5. Store the results
6. Next time onwards use it for prediction (either through GPU or CPU)

But what about the **inaccuracy** part?

Let's dive a little deeper into that.

I ran all the test cases using the GPU matrix multiplication function. Almost all tests that were associated with matrix multiplication failed. I then switched back to the CPU-based matrix multiplication. All tests ended in green tick this time.

## Accepting the Limits
I had already spent more than two days fixing things here and there, integrating CUDA code, and tackling related issues. I saw **two major challenges**: first, if I wanted to gain the speed boost, I'd need to move the whole regression module inside CUDA, otherwise time spent transporting data back and forth between CPU and GPU would eat up all the gains. To do that, I would have to write everything in C, which defeats my initial purpose of learning Rust and Machine Learning as a whole.

The second challenge was that I had become very rusty with the **C language** as I hadn't touched it in almost 16 years. This meant I would have to learn C thoroughly just to write the CUDA code, delaying my original learning journey with Rust and ML.

Hence, I accepted that I wouldn't work on the GPU at that stage. Down the line, if I really feel the need for it, I'll revisit the topic. Until then, I'd be content running small datasets with longer execution times.

Little did I know, the next day would bring one of my biggest facepalm moments in my programming journey...

