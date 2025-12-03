# The Comeback
After failing in my last attempt at integrating the CUDA code into my library, I resorted to the CPU. The logistic regression program was running perfectly fine for a small 100-row, 2-column, dataset. The next logical step was to build a two-layer "toy" neural network.

The neural network doubled the number of matrix multiplications. The logistic regression program was taking around an hour for just 1000 iterations, which already made me impatient. The small two-layer toy neural network took it even further: approximately two hours to run 1000 iterations.

And this was just the beginning, with just two layers. If I actually had to do some complex work, I would have to go beyond two layers and most probably I would need more than one perceptron in each layer, which would contribute to a polynomial growth in the computational complexity of the program.

It felt very frustrating to wait for so long, especially when I could have spent some time and made the GPU work.

The wait time for a decent output was beyond my patience, compelling me to work with smaller datasets, which made all my attempts feel like merely a `Hello World` program. It appeared crystal clear to me that if I were to complete this project, I would definitely have to push my CPU-based, sequential program to a GPU-based, parallel program.

I rolled up my sleeves again to find a solution. 

__If NVIDIA CUDA examples can run on the GPU, the individual Rust Program can run on the GPU, why can't my library program **run** on GPU?__

## Another Pivot
I changed my course of action. I started simply with element-wise addition this time. I wrote a simple program and ran it for 10 elements: it went fine. 

This is the simple CUDA code I used to verify the element-wise addition functionality. It confirmed the basic steps of memory allocation and kernel launch.

```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void add(const float* A, const float* B, float* C, int N) {
	int rowID = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int elemID;											// Element address

	if(rowID < N && colID < N){
		elemID = colID + rowID * N; 				
		C[elemID] = A[elemID] + B[elemID];
	}
}

int main() {
    int n = 10000;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    float size = n;

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<1, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("%f + %f = %f\n", a[i], b[i], c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}


```

I bumped the array size to 100,000,000, and it started returning 0 for many elements. I brought it back to lower numbers. It went fine for 10, 100 and 1000 elements, but started returning 0 when I went for 10000.

This was unusual for me. I put on my debugging hat. The first mistake was in calculating memory. I was making a float array, but I was initializing to $n$ (the element count) instead of $n \times sizeof(\text{float})$..

__Typical type/memory error...__

## The Memory Management
I ran the program again, hoping to get a correct response this time. It did not go well this time either.

Then, there it was: I was launching only one block with 1024 threads, instead of launching sufficient number of threads according to the size of my matrix. With this clue, I changed the initialization as follows -

```c
int threadsPerBlock = 1024;
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
``` 

I ran the program with giant arrays of size 100,000,000. It started giving the correct result instead of returning `0`. The solution was right there.

**Fun time! I played mindlessly with this program until I lost interest in it.**

## The Integration
Once my last toy stopped amusing me, I went on to build a new one. I took another building block - **Rust**, and started integrating my Vector Addition program into my library.

The first step was to separate the CUDA kernel from the main logic. I deleted the main function from the cuda code and ran the following to generate the `ptx` output, which I would consume in the library.

```shell
iron_learn/kernels$ nvcc -c matrix_mul.cu -o matrix_mul.ptx --ptx
```

I also planned to add a new flag in the `main` runnable app in my Rust program to indicate if the program should be run on CPU or on GPU.

After fixing various issues for around 2 hours, I was finally on track, this time with a flag to indicate whether I wanted the program to run on cpu or gpu.

To do this, I had to go through a few steps. First, I created an application context to keep the CUDA context alive throughout the lifetime of the program. CUDA context is slow to initialize, so it is a wise choice to keep the CUDA context alive.

I used Rust's `OnceLock`. This ensures thread safety and prevents race condition - ensuring the global CUDA context was initialized only once. It definitely was a great learning altogether. The application context works as a **singleton** to hold all my application parameters that I wanted to keep alive throughout the entire run.

```Rust
use std::sync::OnceLock;

#[derive(Debug)]
pub struct AppContext {
    pub app_name: &'static str,
    pub version: u32,
    pub gpu_enabled: bool,
    pub context: Option<cust::context::Context>,
}

// Declare a global mutable static
pub static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();

// Initialize the context
pub fn init_context(
    app_name: &'static str,
    version: u32,
    gpu_enabled: bool,
    context: Option<cust::context::Context>,
) {
    let ctx = AppContext {
        app_name,
        version,
        gpu_enabled,
        context,
    };
    GLOBAL_CONTEXT
        .set(ctx)
        .expect("Context can only be initialized once");
}
```

That kept the CUDA context alive, but after doing dozens of changes, the CUDA code wasn't giving me the proper result. I wrote a C program to validate my CUDA code. It was giving correct result.

## The Debugging Hat, One more time 😔

At some point, it struck me: I might be sending data in wrong format. My tensor code was a linear 1D array of data. But the CUDA kernel was treating it as a 2D matrix. With this newfound idea, I started comparing my C code and Rust code.

Few things were off actually:

1. I was sending wrong matrix row columns for calculations. 🤦
2. I was using `f64` (double precision) but my cuda code was expecting `f32` (single precision)
3. The C Program did not complain and worked fine, as it was already taking `float`.
4. What I had to do was to change my CUDA code to standardize the precision to `f64`, instead of `f32`

**Learning:** Precision matters

After successfully integrating the CUDA code in my program, I found it was taking longer than my CPU Code, around 44 seconds for 10 iteration. I had seen this before, it was caused by copying data to device and copying it back to host for each gpu multiplication. So, my CUDA Code may have been performing in microseconds but data transfer was taking longer time, defeating the purpose of vectorization.

I verified my thought process with another log. For 10 iterations, 20 matrix multiplications are running. Around 200 milliseconds are spent on the actual matrix multiplications, while the whole process takes 44 seconds.

So, to get the full benefit, I had to move my whole logic inside the CUDA program.

I wrote few small GPU Modules to run inside cuda

```cuda
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
extern "C" __global__ void matrixMulKernel(double *A, double *B, double *C, int numARows, int numAColumns, int numBColumns)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < numARows && Col < numBColumns)
    {
        double Cvalue = 0.0;

        for (int k = 0; k < numAColumns; ++k)
        {
            Cvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        }

        C[Row * numBColumns + Col] = Cvalue;
    }
}

extern "C" __global__
void sigmoidKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

extern "C" __global__
void vectorSub(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__
void scaleVector(float* v, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] *= scale;
    }
}

extern "C" __global__
void updateWeights(float* w, const float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        w[idx] -= grad[idx];
    }
}
```

I then wrote the orchestration inside rust,

```Rust
pub fn run_ml_cuda() -> cust::error::CudaResult<()> {
    let l: f32 = 0.001; // learning rate
    let e: usize = 5000; // epochs

    let contents = fs::read_to_string("data.json").expect("Failed to read data.json");
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data { logistic, .. } = data;

    let rows = logistic.m as usize;
    let cols = logistic.n as usize;

    // Load PTX module
    let ptx = include_str!("../kernels/gradient_descent.ptx");
    let module = Module::from_ptx(ptx, &[])?;

    // Retrieve kernels
    let matrix_mul = module.get_function("matrixMulKernel")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let vector_sub = module.get_function("vectorSub")?;
    let scale_vec = module.get_function("scaleVector")?;
    let update_w = module.get_function("updateWeights")?;

    // Allocate device buffers
    let d_X = DeviceBuffer::from_slice(&logistic.x)?;
    let d_y = DeviceBuffer::from_slice(&logistic.y)?;
    let d_w = DeviceBuffer::from_slice(&vec![0.0f32; cols])?;
    let d_lines = DeviceBuffer::<f32>::zeroed(rows)?;
    let d_prediction = DeviceBuffer::<f32>::zeroed(rows)?;
    let d_loss = DeviceBuffer::<f32>::zeroed(rows)?;
    let d_grad = DeviceBuffer::<f32>::zeroed(cols)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Kernel launch params
    const TILE: u32 = 32;
    let block2d = (TILE, TILE, 1);
    let grid_x = ((cols as u32) + TILE - 1) / TILE;
    let grid_y = ((rows as u32) + TILE - 1) / TILE;
    let grid2d = (grid_x, grid_y, 1);

    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    let grid_cols = ((cols as u32) + 255) / 256;

    let start = Instant::now();

    // Training loop
    for i in 0..e {
        // 1. lines = X * w
        unsafe {
            launch!(matrix_mul<<<grid2d, block2d, 0, stream>>>(
                d_X.as_device_ptr(),
                d_w.as_device_ptr(),
                d_lines.as_device_ptr(),
                rows as i32,
                cols as i32,
                1i32
            ))?;
        }

        // 2. prediction = sigmoid(lines)
        unsafe {
            launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_lines.as_device_ptr(),
                d_prediction.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 3. loss = prediction - y
        unsafe {
            launch!(vector_sub<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_prediction.as_device_ptr(),
                d_y.as_device_ptr(),
                d_loss.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 4. gradient = X^T * loss
        unsafe {
            launch!(matrix_mul<<<(grid_x,1,1), block2d, 0, stream>>>(
                d_X.as_device_ptr(),
                d_loss.as_device_ptr(),
                d_grad.as_device_ptr(),
                cols as i32,
                rows as i32,
                1i32
            ))?;
        }

        // 5. scale gradient
        unsafe {
            launch!(scale_vec<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_grad.as_device_ptr(),
                l / rows as f32,
                cols as i32
            ))?;
        }

        // 6. update weights
        unsafe {
            launch!(update_w<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_w.as_device_ptr(),
                d_grad.as_device_ptr(),
                cols as i32
            ))?;
        }

        stream.synchronize()?;

        if i % 500 == 0 {
            println!("Iteration {} complete", i);
        }
    }

    let duration = start.elapsed();
    println!("GPU Logistic Regression Training Time: {:.2?}", duration);

    // Copy final weights back
    let mut w_host = vec![0.0f32; cols];
    d_w.copy_to(&mut w_host)?;
    println!("Final weights (first 10): {:?}", &w_host[..10]);

    Ok(())
}
```

And gave it a spin. To my surprise, the whole process took only **11.34 seconds**. I was astonished; the same process took me **1 hour** in CPU. 

**That's the power of parallelism.**

## The Final Nail in the Coffin

Although I integrated the CUDA code in my library, I was disheartened. Speed without accuracy meant nothing. The joy was short-lived when I again ran into NaN...

I did not have to debug much; it was data type mismatch issue again. Everywhere in the CUDA code I wrote $f32$, except for the matrix multiplication inputs. I changed the remaining $f32$ calls to use $f64$ (double precision) for consistency.

After a few fixes and some more code, I finally ran the prediction and it was completing with 11 seconds for sure but with only 7.42% accuracy.

I wore the debugging hat again, for the last time. A few $f32$ needed changing, and a few matrix dimensions were wrongly set. Once all these were fixed, I was able to catch up **54%** accuracy but my **CPU** process gave **92%** accuracy.

Something was still missing. I found that the transpose function was not correctly returning the result. So, I changed the implementation. And the new implementation returned **92.85%** accuracy over **20,000** iterations.

|Metric                       |CPU Performance (Baseline)|GPU Performance (Final Fix)|
|-----------------------------|--------------------------|---------------------------|
|Time (5000 Iterations)       |≈ 1 Hour                  |11.34 seconds              |
|"Accuracy (20,000 Iterations)|92%                       |92.85%                     |

Finally, I could use my **GPU** through my **Rust Library**. The **GPU** actually aided my learning for the first time...
