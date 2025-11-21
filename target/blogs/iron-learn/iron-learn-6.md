I know all my GPU kernels are still not the most highly optimized set. But I really don't care much at this point. I have just enough performance boost that 20000 loops of training runs under 10 seconds, which was taking 1 hour few dayss ago.

I will rather deep dive in learning the next step of machine learning than making my code super efficient.

But one thing, I can do for sure is to write a build script for the GPU Kernels to build automatically.

```
use std::path::PathBuf;
use std::process::Command;
use std::io::{self, Write, ErrorKind};

fn main() {
    // 1. Define the input CUDA source file and the desired output PTX file name.
    let kernel_src = "kernels/gradient_descent.cu";
    let kernel_ptx_name = "gradient_descent.ptx";

    // 2. Tell Cargo which files to watch:
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", kernel_src);

    // 3. Get the output directory path set by Cargo (MANDATORY location for Rust linking).
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join(kernel_ptx_name);
    
    // Attempt to compile CUDA kernel
    let compilation_result = Command::new("nvcc")
        .arg(kernel_src)
        .arg("-o")
        .arg(&ptx_path)
        .args(&["-ptx", "-arch=sm_50", "--allow-unsupported-compiler", "-lcublas"])
        .status();

    match compilation_result {
        Ok(status) => {
            if status.success() {
                // Compilation succeeded, PTX file is in $OUT_DIR
                println!("cargo:warning=Successfully compiled CUDA kernel to: {}", ptx_path.display());

                // --- NEW STEP: Copy the PTX file to the project's kernels/ directory for easy inspection ---
                let kernel_copy_dir = PathBuf::from("kernels");
                let kernel_copy_path = kernel_copy_dir.join(kernel_ptx_name);
                
                // Ensure the 'kernels' directory exists before copying
                if let Err(e) = std::fs::create_dir_all(&kernel_copy_dir) {
                     println!("cargo:warning=Could not create 'kernels/' directory for copy: {}", e);
                }

                match std::fs::copy(&ptx_path, &kernel_copy_path) {
                    Ok(_) => {
                        println!("cargo:warning=Copied PTX file to project root for inspection: {}", kernel_copy_path.display());
                    }
                    Err(e) => {
                        // Log a warning, but don't panic as the build technically succeeded
                        println!("cargo:warning=Failed to copy PTX file to kernels/ directory: {}", e);
                    }
                }
                // ------------------------------------------------------------------------------------------

            } else {
                // nvcc was found, but compilation failed (e.g., syntax error in .cu file)
                panic!(
                    "nvcc was found but failed to compile {}. Exit code: {}. Check your CUDA code.",
                    kernel_src,
                    status.code().unwrap_or(-1)
                );
            }
        }
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                // === GRACEFUL FALLBACK: NVCC NOT FOUND ===
                println!("cargo:warning=NVCC not found ({}). Skipping CUDA compilation and creating an EMPTY placeholder PTX file.", e);
                println!("cargo:warning=Please ensure your main Rust code checks if the included PTX string is empty and falls back to CPU execution.");

                // Create the empty file the main Rust code expects to exist via `include_str!`.
                let mut file = std::fs::File::create(&ptx_path)
                    .unwrap_or_else(|e| panic!("Failed to create placeholder PTX file at {}: {}", ptx_path.display(), e));
                
                writeln!(file, "").unwrap_or_else(|e| panic!("Failed to write to placeholder PTX file at {}: {}", ptx_path.display(), e));
                
            } else {
                // Some other I/O error (permission denied, etc.)
                panic!("Failed to execute nvcc command: {:?}", e);
            }
        }
    }
}
```

I ran it and it went successfully.

Everything now just perfect. It was so satisfying to see GPU getting used by my own program, I could not resist the urge, I bumped up the iteration loop to 10 million epochs. My GPU frequently went 100% usage. Wow!!! A cherishable moment.

10 million iterations took almost 45 minutes in my machine, still less than the time it took for CPU to run only 5000 iterations. Accuracy also hit 93.09% this time.

After playing for some other data and few hyperparameter tuning, I took my next step in the journey. To build a neural network.

At this point my inventory include quite a few things actually.

1. An almost accurate CPU Powered Tensor library
2. A Gradient Descent implementation which runs on CPU
3. Few reusable cuda programs
4. A fully fledged GPU powered Tensor kernels. 
5. A complete orchestrator of Gradient Descent
6. A full logistic regression program which returns accuracy almost identical to other libraries.

While taking a look, I thought of refactoring things a little and also get rid of deprecated warnings.
