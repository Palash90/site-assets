# The Comeback
After failing in my last attempt at integrating the CUDA code into my library, I resorted to the CPU. The logistic regression program was running perfectly fine for a small 100-row, 2-column dataset. The logical next step was to build a two-layer neural network.

The neural network doubled the number of matrix multiplications. The logistic regression program was taking around an hour for just 1000 iterations, which already made me impatient. The small two-layer toy neural network took it even further: approximately two hours to run 1000 iterations.

And this was just the beginning, with just two layers. If I actually had to do some complex work, I would have to go beyond two layers and most probably I would need more than one perceptron in each layer and that would contribute to a polynomial growth in the computational complexity of the program.

It felt very frustrating to wait for so long, especially when I could have spent some time and made the GPU work.

The wait time for a decent output was beyond my patience, compelling me to work with smaller datasets, which made all my attempts feel like merely a `Hello World` program. It appeared crystal clear to me that if I were to complete this project, I would definitely have to push my CPU-based, sequential program to a GPU-based, parallel program.

I rolled up my sleeves again to find a solution. 

__If NVIDIA CUDA examples can run on the GPU, the individual Rust Program can run on the GPU, why can't my library program **run** on GPU?__

## Another Pivot
I changed my course of action. I started simply with element-wise addition this time. I wrote a simple program and ran it for 10 elements: it went fine. 

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

This was unusual for me. I put on my debugging hat. The first mistake was in calculating memory. I was making a float array, but I was initializing to `n`(the element count) instead of $n \times sizeof(\text{float})$..

__Typical type/memory error...__

## The Memory Management
I again ran the program hoping to get a correct response this time. It did not go well this time too.

Then, there it was, I was allocating only one block with 1024, instead of allocating memory according to the size of my matrix. Changed it as follows

```c
int threadsPerBlock = 1024;
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
``` 

Then it started working. I got it running in C. I had to get it running in Rust.

The first step is to separate out the cuda code and the C code. I deleted the main function from the cuda code and ran the following to get the ptx output, which we'll feed to our Rust code.

```shell
iron_learn/kernels$ nvcc -c matrix_mul.cu -o matrix_mul.ptx --ptx
```

I also added a flag in the `main` runnable app in my rust program to indicate if the program should be run on CPU or on GPU.

After fixing various issues for around 2 hours, I finally was able to again back on track, this time with a flag to indicate, if I want the program to run on cpu or gpu.

To do this, I had to go through few steps. First, I created a Application Context to keep the context alive and it was a great learning altogether. Now, the appcontext works as a singleton to hold all my application parameters that I want to keep alive throughout the application.

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

After doing dozens of changes, still cuda code was giving me proper result. I wrote a C program to validate my CUDA code. It was giving correct result too.

Then it struck me, I may be sending data in wrong format. My tensor code was all in a single vector, a linear 1D array of data. But CUDA expects it in a matrix format. With this new found idea, I started comparing my C code and Rust code. Few things were wrong.

1. I was sending wrong matrix row columns for calculations
2. I was using `f64` but my cuda code was expecting float
3. That's why my C Program did not complain and worked fine, as it was already taking floating point.
4. So, now what I have to do change my cuda to expect f64

Learning- Precision matters

After successfully integrating the CUDA code in my program, I found it is now taking longer than my CPU Code, around 44 seconds for 10 iteration. I had seen this before, it is caused by copy to device and reverse copy to host for each gpu multiplication. So, my CUDA Code may be performing in microseconds but copying is taking time.

I verified my thought process with another log. For 10 iterations, 20 matrix multiplications are running. Around 200 milliseconds are spent for the actual matrix multiplications, while the process takes 44 seconds.

So, to get the full benefit, I have to move my whole logic inside the cuda program.

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
        float Cvalue = 0.0;

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

And gave it a spin. To my surprise, the whole process took only 11.34 seconds. I was astonished, the same process took me 1 hour in CPU. 

That's the power of parallelism.

Anyways, I was heart broken too, I again ran into NaN...

Did not have to debug much, it was data type mismatch issue again. Everything in CUDA I wrote was float, except for the matrix multiplication. I changed all of those to double.

After few fixes and some more code, I finally ran the prediction and it was completing with 11 seconds for sure but with only 7.42% accuracy.

Wore the debugging hat again. Few f32 needed change, few matrix dimensions were wrongly set. Once all these fixed, I was able to catch up 54% but my CPU process gave 92% accuracy.

Something is missing. I found that the transpose function was not correctly returning result. So, I changed it. And the new implementation returned 92.85% accuracy with 20000 iterations.


Finally, I could use my GPU inside my Rust Library. The GPU actually aided my learning for the first time...
