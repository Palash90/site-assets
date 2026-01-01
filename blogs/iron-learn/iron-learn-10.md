# The Grand Finale: The Full Image Reconstruction Network from Scratch in Rust

The next day I checked things started evolving properly but slowly. There was no improvement in execution time. It got stuck in that 34 - 37 second time frame. My gut feeling told me, destination is not far. However, when I opened my IDE for debugging, it was not looking so. It was again a mess after all the new code amendments and new experimentations. It was getting tougher to debug without a nice looking code base.

So I decided on end to end refactoring and  code clean up.

Oh boy!!!

## Boredom Brings Solution
I just found my answer in the boredom of refactoring. Suddenly it crossed my mind that 64 bit floating points (`f64`) might be the bottleneck in my `Tensor` Library. To confirm my theory, I switched to the Python script, I switched every array to `float64`, instead of `float32` and ran it. This time, it started taking more than 50 seconds, far more than my Rust Tensor program. 

Yes, I cracked it. It was not Rust, Python, `cupy` or GPU that was making the difference in the execution time, it was the data type. The holy grail of low level programming.

I switched back to my rust program to fix the remaining clean up, refactoring and build issues.

## The Rust Type System
Well, I found the issue and now I have to implement the fix in Rust. I was in the middle of the refactoring. I already separated the network layer, the builder and the loss functions. While doing so I struggled to fix the Generics Issues. I managed to work around those.

But final blow came to it when I found that, the neural network I have built heavily relies on floating point mathematics. Especially, gradient descent, learning rate, scaling and literally every math operation works on floating point mathematics. I had a vague idea on how to solve the problem. I do all my maths in floating point and then round of the final result to Integer.

But just in point, out of curiosity, I started finding about it, how Integer neural networks work. I came to an idea that, my methodology works but instead of rounding, some other idea quantization is at play. I did not bother to look for it. Will know in the due course of time.

At that point, my main motto was to fix the build issues, generics issues and make my program run again. And probably, use `f32` instead of `f64`.

## The Illegal Access

I made all the fixes and built the code. The build was successful and I started running the program. Execution passed 30 seconds, my hope started building up and just then the tensor program started failing with `IllegalMemoryAddress` and forward pass resulted in `NaN` once again.

## The Great Demotivation
That run time error tipped me off my threshold. It was enough. I spent enough time and resources to fix everything. I already achieved what I wanted this project to give me. I learnt Rust, fought with the compiler, worked with FFI, invoked external device, wrote CUDA Programs, learnt Machhine Learning and Deep Learning, successfully wrote a neural network, learnt about SIREN. This whole experience has set me to a better Rustacean and a more knowledgable ML Hobbyist path than I was 2 years ago.

I accepted my fault in the plan, Iron Learn was too ambitious as a goal to pursue, that too single handedly. I thought, it would be better if I shut it down at that point. 

I was ready to accept: 'No more Iron Learn'.

I made a "Sunset" plan and visited Google Graveyard to console myself "Look, Google failed too, that big giant made a mistake too and most of them are older than your 2 year project. Don't mourn, don't take it to your heart and just do it".

I came back with a heavy heart, consulted Gemini and ChatGPT to prepare a mourning speech that I will post in the `README.md` and called it a day.

## The Drama King
Next day morning I opened the IDE to write the eulogy. Started playing "Yaariyaan Reprise from Cocktail", one of my favourite emotional song. Shed a few drops of tears. I know it's nothing to the outside world but Iron Learn was one of the finest project I have ever worked with in my 18 years of acquiantance with computers.

Again, don't judge me. I do things at times that make no sense at all or makes the most sense that I don't understand at all.

In bengali, we have a popular proverb - রাখে হরি মারে কে| That exactly what happened to `Iron Learn`.

I consulted the AI friends and came up with an eulogy for Iron Learn. Started writing the first line.

The same inner voice told me, "Why not give it a last spin?"

I ignored and started writing the second line. "You are being unreasonable here".

Again ignored the inner voic and started the third line. "I promise you, if not solved within 1 hour, I will let you complete the eulogy".

Ok fine. Let's take the final spin.

## Defying the Flatline
I cleared up everything I wrote, I discarded the changes in the `README.md` and started finding the root cause of the memory error. It was right there in the Kernel programs. I switched everything to `f32` in the Rust program but my kernels were still referrring to `double`, a change I made back in the days of Logistic Regression when everything I wrote was in `f64`.

Basically, for every allocated byte block, I was trying to read twice as much and was running into the protected memory space. The GPU version of `segfault`.

After the fix, things started running magically again and I fixed it under 45 minutes. My Iron friend came back from Coma. It was alive again.

## The NaN Fix
After the fixing the memory access, I still had the second one to tackle too.

This one was actually easy and kind of known to me too. I was not having a `clip` function in either of my CPU or GPU tensors and at that time, I was using Binary Cross Entropy function which can result in a NaN due to a log function in use. Log of negatives result in NaN.

I implemented the `clip`:
```rust
let result = data
            .iter()
            .map(|a| {
                if *a < min {
                    min
                } else if *a > max {
                    max
                } else {
                    *a
                }
            })
            .collect();
```

```c
extern "C" __global__ void clip(const float *s, float *r, int n, float min, float max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (s[idx] < min)
        {
            r[idx] = min;
        }
        else if (s[idx] > max)
        {
            r[idx] = max;
        }
        else
        {
            r[idx] = s[idx];
        }
    }
}
```

The NaN also vanished just like that. Two critical fixes within 90 minutes.

## The Void
The next run of the program I could complete the network training and could generate few images. At first everything looked just the same, a full white Rectangle indicating that something is off.

I put some logs and it brought my old friend - 'Normalization/Denormalization' pair. This time, lack of normalization was the issue. Once reintroduced, things went smooth. 

## The 5 Hours of Suspense
Finally Iron Learn was ready. I took it for a spin. I spent another 5 hours training the network, this time with more layer (128 layers). Built a 99k parameter image model and here you go with the results:

### The Static Network Started With
![Static](${iron-learn-10-output0})

### The Final 200×200 Reconstruction
![Final200*200](${iron-learn-10-output803000})

It is satisfying to watch a neural net learn and here is what I want you to take a look at too:
### The Timeline
![Time Lapse](${iron-learn-10-timeline})

I did not stop just at reconstruction, I tried with different sizes too. Here is one 512×512 reconstruction.
### The Reconstruction on 512×512
![Final512*512](${iron-learn-10-512})


