# Fighting the Rust Compiler Stage

The moment, I put the cust code in my program, it started giving me angry red eyes and screamed at me with multiple errors. The compiler correctly pointed out that `DeviceCopy` trait from cust library has not been implemented for my Tensor type.

Ah, the classic trait bound error which I almost forgot after working in python and JS for last 14 months. Rust is so secure, it won't let me play with memory carelessly. Well, the `cust` library took a step forward and made this even harder for any types which refers to raw pointers. `Vec<u32>` and `Vec<T>` are obviously one of those and these are the backbone of my Tensor


