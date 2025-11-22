# Accelerating Calculations: From CPU to GPU with Rust and CUDA
The success of the logistic regression gave me a confidence boost, but the program's performance felt like a punch to the gut. The naive matrix multiplication module was taking forever to complete even on a small dataset like (321 * 6). No programmer enjoys staring at the console for the program to finish its execution.

I have a CUDA-enabled GPU sitting in my machine gathering dust. I had avoided configuring it for development for two years out of sheer laziness. But, to scale up I had to conquer my laziness. I had to make my GPU perform the heavy math.

I spent some time searching alternatives. Many search results pointed to `ndarray`, a high-performance rust library to work with multi-dimensional arrays. It's a great library, but using it out of the box felt like cheating, like another black box "magic" to me. My motive was to learn from scratch; any ready-made solution would have defeated that purpose. I stumbled upon `rust-cuda`, a toolkit for integrating GPU programming into Rust.

Another rabbit hole opened up...

## The Setup
To unlock the potential of those CUDA cores, the first step was setting up the environment. And on Windows, that was not an easy task. The setup was the single most time-consuming part of the whole process.

I spent 3 to 4 hours just to see my GPU actually doing something.
- I installed CUDA Toolkit
- I installed cmake
- I installed MSVC 19
- I installed Visual C++ build tools
- I installed Microsoft Visual Studio
- Finally I was able to run the cuBLAS matrix multiplication program

Phew!!!

Initial setup completed. The effort was worth it. I spent an hour playing with the CUDA Sample programs, just for the fun of it. Multiplying a 1280 * 960 matrix by a 960 * 640 one took just 0.619 msec and the GPU did not even break a sweat. I tried to go even higher - multiplying a 12800 * 9600 matrix by 9600 * 6400 matrix. I was amazed by the performance. It took only 392.912 msec.

For the first time in two years, I finally used my NVIDIA GPU the exact way I always intended to use it.

## Unveiling the Secret
I was curious to know how the GPU delivers that speed. The high level answer was known - parallelization - Single Instruction Multiple Data. However, I was not sure of the mechanism. I dove deep in CUDA architecture resources including the Programming Guide, NVIDIA Developers youtube channel and many other youtube videos.

**Disclaimer**: I am not a hardware major and the following is a very vague understanding. I may be completely wrong on the specifics.

The GPU (device) is treated like an external device which communicates with CPU (host). If the CPU has to use the device, it first needs to get the device to first allocate memory for it. Once memory is successfully allocated on the device, the data transfer happens. Then the host asks the device to perform the intended operation. The host then passes a reference to the compiled kernel routine (the function that runs on the GPU) for the device to execute. This is done via a kernel launch. Once the kernel launch succeeds, the memory is read back from device to host.


```
+----------------+                                 +----------------+
|      HOST      |                                 |     DEVICE     |
|    (CPU Code)  |                                 |    (GPU Code)  |
+----------------+                                 +----------------+
        |                                                    ^
        | 1. Allocate Device Memory (cudaMalloc)             |
        |--------------------------------------------------->|
        |                                                    | (d_a, d_b, d_c pointers returned)
        | 2. Copy Data Host -> Device (cudaMemcpy)           |
        |      (e.g., a -> d_a, b -> d_b)                    |
        |--------------------------------------------------->|
        |                                                    |
        | 3. Launch Kernel                                   |
        |      (vectorSub<<<grid, block>>>(d_a, d_b, d_c, n))|
        |--------------------------------------------------->|
        |                                                    | 4. Kernel Execution
        | (Host can do other work here, if asynchronous)     |    - Multiple threads execute vectorSub
        |                                                    |    - Accessing d_a, d_b, writing to d_c
        |                                                    |<----------------------------------+
        |                                                    |                                   |
        | 5. Synchronize (cudaDeviceSynchronize)             |                                   |
        |<---------------------------------------------------|                                   |
        |                                                    | (Kernel completes, results in d_c)
        | 6. Copy Data Device -> Host (cudaMemcpy)           |
        |      (e.g., d_c => c)                              |        
        |<---------------------------------------------------|
        |                                                    |
        | 7. Free Device Memory (cudaFree)                   |
        |--------------------------------------------------->|
        |                                                    |
+----------------+                                 +----------------+
```

The GPU essentially manages a hierarchy of threads - Grids, Thread Blocks and Threads. Each thread executes the same kernel on different data. GPUs have the capability of launching millions of threads per second. This is where the speed comes from. The parallelization is done by the GPU based on the launch parameters like number of threads, number of blocks etc. set by the programmer.

The host launches a kernel to this massive system of threads, each thread runs the same operation. As long as each thread can work independently of others, we can run them simultaneously.

```
+----------------------------+
|          Grid              |
|  +---------------------+   |
|  |   Thread Block 0    |   |
|  |  [T0][T1][T2][T3]   |   |
|  +---------------------+   |
|  +---------------------+   |
|  |   Thread Block 1    |   |
|  |  [T0][T1][T2][T3]   |   |
|  +---------------------+   |
|  ...                       |
+----------------------------+
```

## The Rust Integration Attempt
When I got bored with playing around, I took a step forward. This was the time to look for some advanced Rust stuff. There is no native library for CUDA programming provided by NVIDIA. So, I had to wire the bindings on my own. I spun up a new small project for the POC.

I started with `rust-cuda`. After spending hours on it without any success, I resorted to other libraries. A quick Google search gave me a few other options. I tried `cust`, `rust-gpu` and some other random internet suggestions. `cust` worked. 

I wrote my first ever CUDA program - a simple vector subtraction.

```c
#include <cuda.h>
#include <cuda_runtime.h>


extern "C" __global__
void vectorSub(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}
```

```rust
// Load compiled CUDA kernels from PTX module
let ptx = include_str!("../kernels/vector_sub.ptx");
let module = Module::from_ptx(ptx, &[])?;

// Retrieve kernel function references
let sub = module.get_function("vectorSub")?;

// Allocate GPU memory buffers
let d_a = DeviceBuffer::from_slice(&a)?; 	// First input vector: rows
let d_b = DeviceBuffer::from_slice(&b); 		// Second input vector: rows
let d_c = DeviceBuffer::from_slice(&vec![0.0f32; rows])?; // Result vector

unsafe {
        launch!(sub<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            rows as i32
        ))?;
    }

d_c.copy_to(&mut c)?;
```

The successful run and a correct result indicated that the Rust FFI to execute the hardware kernel actually worked. Again with another confidence boost, I started writing the next program - a naive matrix multiplication.

```c
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
extern "C" __global__ void matrix_mul(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < numARows && Col < numBColumns)
    {
        float Cvalue = 0.0;

        for (int k = 0; k < numAColumns; ++k)
        {
            Cvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        }

        C[Row * numBColumns + Col] = Cvalue;
    }
}
```
Started small: multiplying a 2 * 1 matrix by 1 * 2 matrix. I validated these results against CPU computed results. Next, I multiplied two matrices filled with random numbers and validated results and performed a few more test runs.

Once these small matrices were done, it was time to go bigger. Multiplying a big matrix of 1024 * 1024 by another 1024 * 1024 took only a few milliseconds to complete.

## Conclusion
Finally, I did it. I overcame my laziness, put in the effort and achieved what I planned 2 years ago. I was very excited to integrate this setup into my main Machine Learning repo.

Only if I knew that a nightmare was waiting to happen...

