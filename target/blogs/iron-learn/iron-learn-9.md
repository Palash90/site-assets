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
No matter how small learning rate I chose, the training always was getting stuck into the error oscillation loop. At one point, it occurred to me that, if I choose very small learning rate like 0.000001 I could save the training but definitely, that would take me longer. Then I thought, what if I can gradually decrease the learning rate programatically. I did some research on my thoughts and found about Cosine Annealing. I applied the cos learning decay function and it started showing smooth learning.

```python
decay_factor = 0.5 * (1 + math.cos(math.pi * i / (epochs+epoch_offset)))
current_lr = lr_min + (lr_max - lr_min) * decay_factor 
```

## The Result: Art through Math
After fixing all the issues and running the network for almost 1.2 million iterations (around 4 hours on my machine), I could see generated output very close to the input image. The input was a very complex high-dimensional function, far beyond simple XOR gate test set or even the logistic regression dataset. This proves that my neural network was exhibiting properties of Universal Approximation Theorem. I can now use this network to do bigger things.

For comparison, here are the results:
### Original Image
![Original Image](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-original.png "Original Image")

### The Initial Static
![Initial Static](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-plot_static.png)

### Final Image on 200x200
![Final Reconstruction](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-plot_final.png)

### Reconstruction on Higher Resolutions
At that point, I was pretty sure, the network learnt the underlying function. With that confidence and the saved weights, I tried to test it against different blank canvas sizes like 512, 50, 1024 etc. In every blank canvas it drew the image. 

Following is the same image reconstructed on 1024x1024 resolution:

![Higher Resolution Reconstruction](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-9-plot_FINAL_1024.png)

Since the network had learned the mathematical concept of the lines rather than just the pixels, the 1024x1024 version didn't look 'pixelated' like a standard zoom—it looked like the network was redrawing its own masterpiece on a bigger canvas.

## Validation and Next Steps
I basically made a highly complex, inefficient, uber expensive image scaler. The result was satisfying but not perfect. It proved the point but I needed perfection.

I posted the resultant image on Reddit and another redditor commented about `SIREN` or SInusoidal REpresentation Networks.

`SIREN` is a neural network which uses Sine activation function instead of ReLU or Tanh. Mostly used for purpose of Implicit Neural Representation, a technique very similar to what I was trying to achieve. `SIREN` is more effective than other neural networks in representing Images, Audios etc.

I implemented a `SIREN` in python with the hope of reconstructing the image to a more perfect one. But my efforts were in vain, it did not work. I finally abandoned the plan of writing `SIREN` after few failures to pursue something else.

While my network used Sigmoid, SIREN uses Sine waves, which are naturally better at capturing the 'sharp edges' of a drawing. I eventually moved on from SIREN after a few failed attempts, but the experience changed how I looked at the 'frequency' of my data..

In my next attempt, I actually achieved a sharper, more detailed reconstruction.

## The Rust Comeback
The idea I implemented in Python showed some fruiful results. The success of the Python script rejuvenated me. I was ready to take the next challenge. I braced myself to pour some energy into the Rust program. I would have missed the adventure and the learning if I did not come back to Rust.

The journey again resumed which were paused for few days. I wore the Rustacean hat and it took me another week to put everything in place:
1. I wrote a separate `Tensor` trait and put all the defined methods in it
2. I wrote a `CpuTensor` struct and the `Tensor` trait implementation for it
3. I wrote a `GpuTensor` struct and the `Tensor` trait implementation for it

## The Initial Shock
I was expecting my Rust program to work seamlessly out of the box. Then came the next shock. GPU Tensor was taking 90+ seconds to run the same network, which my python program was taking only 8.

Another challenge to solve. Another debugging session.

I tried to find the reason. The Rust code showed nothing, except that every `Tensor` operation was taking a long time to compute. I was very surprised. I doubted my CUDA Kernel programs and used `nsys` profiler. 

The result was a surprise for me, the major time consuming part of my application was not the CUDA Kernels but the memory allocation and deallocation.

Then I tried to reason it with `cupy`. It also needs memory to perfom its operations, then how is it so fast?

The answer lay in the `Memory Pool`. The library does not depend on default memory allocator, rather it uses a custom memory pool

A logical approach for me was be to look for a memory pool, but I found no direct memory pool implementation in the cust library. I first thought of implementing my own but it would be very painful and error prone. I kept on looking for a solution. I finally found it. The `cust` library might not have a memory pool wrapper but CUDA library did so and `cust` library re-exports those modules under `sys`. It was a huge relief for me. But still it was far from easy to implement. 

I tried incorporating the memory pool. I had to do a lot of consultaion with the documentation to finally make it work. 

Here is a snippet of the code:
```rust
pub fn get_mem_pool() -> CudaMemoryPool {
        let device = Device::get_device(0).unwrap();

        // Create a memory pool for the device
        let mut pool = std::ptr::null_mut();
        let pool_props = CUmemPoolProps {
            allocType: cust::sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: cust::sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: cust::sys::CUmemLocation {
                type_: cust::sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            win32SecurityAttributes: std::ptr::null_mut(),
            reserved: [0u8; 64],
        };

        unsafe { cuMemPoolCreate(&mut pool, &pool_props) };

        let reserve_size: usize = 2048 * 1024 * 1024;
        let mut reserve_ptr: CUdeviceptr = 0;
        unsafe {
            // This is often a synchronous call initially, but it gets the memory from the driver
            // and makes it available to the pool.
            cuMemAllocFromPoolAsync(
                &mut reserve_ptr,
                reserve_size,
                pool,
                std::ptr::null_mut(), // Null stream is okay for one-time setup
            );
            // You MUST synchronize the null stream here to ensure memory is available
            cuStreamSynchronize(std::ptr::null_mut());

            // Now free it back to the pool immediately for reuse
            cuMemFreeAsync(reserve_ptr, std::ptr::null_mut());
            cuStreamSynchronize(std::ptr::null_mut());
        }

        println!("Memory pool created for device {}", device.name().unwrap());

        CudaMemoryPool {
            pool: Arc::new(Mutex::new(UnsafeCudaMemPoolHandle(pool))),
        }
}
```
Once it was done in a `main` program outside of my `Tensor`, I could see thousands of memory blocks allocated and deallocated in milliseconds. Of course, it had an initial price to pay for the Memory Pool creation but in most cases, it would be a one time setup cost.

## Integration in GpuTensor
I took the code and put inside my library.

At that point, I was wise enough to not believe it would work on the first try. Unsurprisingly, when I incorporated the code in my Tensor library, it did not work. It was still taking 90+ seconds. I suspected the lifetime of the pool. So I decided to keep the memory pool alive just like I did for context.

Copy, Paste...

And...

Compiler Error.

To solve the issue, I had to use `Arc<Mutex<>>`. And the memory pool was kept alive throughout the application uptime. However, this did not solve the problem. The actual issue was somewhat deeply nested in the CUDA wrappers. The wrappers themselves were using the default allocator and  not the pool.

And I opened another Pandora's Box.

I had to deal with raw pointers. It was the first time, I actually worked with Raw Pointer in Rust. It was a very scary experience but I survived it and wrote a custom device buffer to link between Host Memory and Device Memory:

```rust
impl<T: Numeric + DeviceCopy> Drop for CustomDeviceBuffer<T> {
    fn drop(&mut self) {
        let pool = match &GPU_CONTEXT.get().expect("No GPU Context Set").pool {
            Some(p) => p,
            None => panic!("Cuda not initialized or Gpu Pool is not set up"),
        };

        let _ = pool.free(self.as_device_ptr().as_raw());
    }
}

pub fn get_device_buffer<T: Numeric + DeviceCopy>(size: usize) -> CustomDeviceBuffer<T> {
    let pool = match &GPU_CONTEXT.get().expect("No GPU Context Set").pool {
        Some(p) => p,
        None => panic!("Cuda not initialized or Gpu Pool is not set up"),
    };

    let ptr_size = size.checked_mul(size_of::<T>()).unwrap();

    if ptr_size == 0 {
        panic!("Attempted a zero size pointer or null pointer creation.");
    }

    let pool_allocated_pointer = pool.allocate(ptr_size).unwrap();
    let device_pointer = DevicePointer::from_raw(pool_allocated_pointer);

    let device_buffer = unsafe { DeviceBuffer::from_raw_parts(device_pointer, size) };

    CustomDeviceBuffer { device_buffer }
}
```

## The Raw Pointer Size Issue
The deal with raw pointers gave me access to low level memory management and fast implementation but I was thrashed by pointer size mismatch issue. I dug into the code of `cust` wrappers. I found the issue, I was allocating for the length of the array but conveniently forgot to account for the data type.

What I was doing is essentially this, for an array of  `f32` of length 1, I was allocating 1 byte of memory in the GPU rather than 4 bytes. I fixed the mismatch and hoped high that this version will work. It worked but execution time did not go down.

I was literally frustrated and thought of abandoning the project creeped up. I called it a day.

## The Hidden Bug
The next morning, with a fresh mind, I started debugging with `nsys`. It was very hard to point out. After a whole day's worth of effort, I finally found the hidden issue: `Tensor::ones`. To calculate `Sigmoid`, the network needed a tensor of 1s. To achieve this, my implmentation was creating a `Vec<f32>` with `1`s and was transfering the `Vec` to the Device, making a H2D copy at every epoch.

I wrote a new kernel to initiate memory in GPU with provided value:

```c
extern "C" __global__ void fill_value(float *out, int n, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = value;
    }
}
```
This solved the issue of H2D copy and brought down the execution time to 54 seconds, far from 8 seconds.

## Another Costly Operation
I was on the lookout for the issue, again took help from `nsys` profiler. This time it showed D2H copy. I found that `sum` reduction function (similar to `np.sum()`) was behind those copies. As `sum` function is an aggregate function and GPU works on a thread based execution principle, initially I thought of doing this calculation in CPU but that backfired heavily. This function gets called on each epoch for loss calculation. A drop in even a few milliseconds would bring down seconds on training for 1000 epochs.

So I wrote a column based reducer instead:

```c
extern "C" __global__ void column_reduce(const float *inputMatrix, float *outputSums, int numRows, int numCols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < numCols)
    {
        float sum = 0.0f;

        for (int row = 0; row < numRows; ++row)
        {
            sum += inputMatrix[row * numCols + col];
        }

        outputSums[col] = sum;
    }
}
```

This cut down some. From 54 seconds, this change brought down to 45 seconds.

## The Matrix Multiplication Refactoring
I never stopped taking help from `nsys`. This time it indicated `cuLaunch` performance issue. The hand-written kernels were making a mess. The usual suspect was the Matrix Multiplication kernel. I replaced the tiled matrix multiplication routine to older thread based non-tiled routine. Not much change, dropped from ~45 seconds to ~34 seconds. Still not hitting the target. I changed some other kernels. Not much change, so reverted all these and settled down with whatever I had prior to those changes.

However, the lag was real and I could not sit quietly on it. I was getting closer. So I tried to use `cuBLAS`. The cust library does not support `cuBLAS`. I had to resort to `cublas-sys` to get cuBLAS working. The road was not smooth. However, after 2 hours of hiccups, I finally managed to integrate cuBLAS.

The following snippet got the job done:
```rust
fn _gpu_mul_cublas(&self, rhs: &Self) -> Result<Self, String> {
        let m = self.shape[0] as i32;
        let k = self.shape[1] as i32;
        let n = rhs.shape[1] as i32;

        let total_elements = (m * n) as usize;
        let result = get_device_buffer(total_elements);

        let alpha = T::one();
        let beta = T::zero();

        unsafe {
            cublasSgemm_v2(
                Self::_get_cublas_handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha.f32(),
                rhs.device_buffer.as_device_ptr().as_raw() as *const f32,
                n,
                self.device_buffer.as_device_ptr().as_raw() as *const f32,
                k,
                &beta.f32(),
                result.as_device_ptr().as_raw() as *mut f32,
                n,
            );
        }

        let result_shape = vec![self.shape[0], rhs.shape[1]];
        Ok(Self::_with_device_buffer(result_shape, result))
}
```

However, I was heartbroken initially as cuBLAS Version took almost the same time and sometimes more than my naive implementation. I tried to profile. Nothing suspicious but not much change either.

Then I started reading about it. cuBLAS works best with really big matrices, which is not the case with the XOR test dataset. Fine for me. It works and I will definitely benefit when I tackle the much larger image reconstruction task.

The story did not end here but energy definitely was taking a hit at that point. It was another two weeks passed already after I resumed working on Rust Tensor Library and still the Simba image reconstruction seemed a far fetched dream. 

