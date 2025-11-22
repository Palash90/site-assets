# Moving the calculations from CPU to GPU
The success of the logistic regression program pushed me to do more. However, my matrix multiplication method was taking forever to complete even for a small matrix like 321 * 6. I had to wait for long time to see things working.

I have a CUDA GPU, I was just feeling lazy to configure it for programming for 2 years. However to progress I had to overcame my laziness. I started looking at internet to get my GPU to do the heavy math.

Many resources cited `ndarray`, a rust library to work with multi-dimensional arrays. The thought of doing everything from scratch discarded that idea however. Again back on searching track. I stumbled upon `rust-cuda`, toolkit to work with GPU.

Another rabbit hole opened up...

## The Setup
To get CUDA cores to work with my program, the very first thing I needed was to actually setup CUDA on my machine. I had the GPU for quite some time and I used it mostly for some video editing purposes. I never set it up for programming.

This was the biggest time consuming journey in the whole process.

I spent 3 to 4 hours just to see my GPU actually doing something.
- I installed CUDA Toolkit
- I installed cmake
- I installed MSVC 19
- I installed Visual C++ build tools
- I installed Microsoft Visual Studio
- Finally I was able to run the cuBLAS matrix multiplication program

The effort was worth it. I played around the CUDA Sample programs for some time, just for the fun of it. A 1280 * 960 vs 960 * 640 matrix multiplication just took 0.619 msec and the GPU did not even break a sweat. I tried to hit even higher numbers - 12800 * 9600 vs 9600 * 6400 matrix multiplication

I was amazed by the performance. It took only 392.912 msec.

For the first time in 2 years, I am using my nvidia GPU the exact way I initially thought I will use it for.

## Unveiling the Secret
I was curious to know how the GPU works so fast. The answer was kind of known to me that it parallelizes the data on different calculation unit. However, I was not sure of the mechanism. I read about it to get some idea.

The following is a very vague understanding I got for now. I may be completely wrong on this one.

The GPU (device) is treated like an external device which communicates with CPU (host). If the CPU has to do anything with it, it first needs to get the device to first allocate memory for it. Once memory is successfully allocated on the device, the data transfer happens. Then the host asks the device to perform the intended operation. The host even can share the external routine calls too for the device to run. This is done via a kernel module compilation. Once the kernel launch succeeds, the memory is read back from device to host.


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
        |      (e.g., d_c -> c)                    		     |
        |<---------------------------------------------------|
        |                                                    |
        | 7. Free Device Memory (cudaFree)                   |
        |--------------------------------------------------->|
        |                                                    |
+----------------+                                 +----------------+
```

**Disclaimer**: I am not a hardware major, so take it with a pinch of salt.

## The Rust Integration Attempt
When I got bored with playing around, the next step started. This was the time to look for some advanced Rust stuff. There is no native library 

I started searching how to write Rust a program that runs on GPU.

I started with rust-cuda. It was not smooth in the first attempt, after spending hours on it and no success, I resorted to other libraries.

A quick google search gave me few other options. I tried `cust`, `rust-gpu` and some other random internet suggestions. `cust` worked. My first ever program was a simple vector subtraction.

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

This went smooth. Then switched to matrix multiplication.

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
Started small with 2 * 1 vs 1 * 2 with small matrices. I validated these results against CPU calculated results. Next, ran against random numbers.

Once these small matrices were done, it was time for going bigger. Running against big set - 1024 * 1024 vs 1024 * 1024 - took few milliseconds to complete.

## Conclustion
Finally, I could do it. I overcame my laziness, I put the effort and I was seeing what I planned 2 years ago. I was very excited to this setup into my actual repo.

Only if I knew that a nightmare is on waiting to happen...

