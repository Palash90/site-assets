# Build Your Own Neural Network from Scratch in Rust: From Zero to Image Reconstruction

## Prologue
Machine Learning often felt like a "black box" to me. Every time I started learning it, I was introduced to `Numpy` at the very least. Libraries like `scikit-learn`, `PyTorch`, `TensorFlow`, etc. are excellent for building quick prototypes as well as production-grade models. But they heavily obscure the underlying mechanics.

Hence, I decided to start learning this technology by building it from scratch. After experimenting with various implementations over the years, I found the missing link: the difficulty lay not in the language, but in the lack of a motivating end-goal.

This project began as a month-long deep dive into Linear Regression. However, my momentum gradually slowed and after numerous iterations, I have finally reached a milestone where I can document this journey. As of this writing, I am still building on it.

## Modus Operandi
This series is designed with a specific philosophy in mind: **Radical Transparency**. We do not start with frameworks or pre-built third-party libraries (we do not even install any of them). We start with a blank file and a single `fn main()`. From there, we will incrementally build the mathematical and structural architecture required to perform complex tasks.

### The Roadmap: From Zero to Image Reconstruction

- **The Blank Canvas:** Initializing the environment and establishing the foundational data structures.
- **The Mathematical Engine:** Implementing tensors, gradients, and backpropagation from scratch.
- **Building the Network:** Constructing layers and activation functions.
- **The Visual Goal:** Training our library to interpret and reconstruct images, proving that 'magic' is just high-dimensional calculus written in a language that doesn't forgive.

![Image Reconstruction]()

**Note to the Reader:** This is the roadmap I wish I had two years ago. Whether you are a Rustacean curious about AI or an ML practitioner looking to understand systems-level implementation, this journey is for you.

And that’s where the story begins...

## The Tensor
To build a neural network from scratch, we need to build its building block first. In the world of Machine Learning, that building block would be a **Tensor**. In simple terms, a tensor is a collection of numbers, organized in a grid.

###  Journey from Scalar to Tensor
To understand the data structure we would be building, we first need an intuition. Let's start building it from scratch as well.

- **Scalar:** We are familiar with this and use it every time we perform arithmetic operations: a single number (e.g. 5). 
    In the world of tensors, we would define them as tensor of rank 0.
    In programming, this would be a single numeric variable: `x=5`
- **Vector:** When we arrange a collection of numbers, we get a `Vector`.
    In the world of tensors, we would define them as tensor of rank 1.
    In programming, this would be an array or `Vec` of numeric variables: `a = [1, 2, 3]`
- **Matrix:** When we arrange multiple vectors in an array, we get matrix.
    In the world of tensors, we would define them as tensor of rank 2.
    In programming, this would be an array of arrays (or `Vec` of `Vec`s): `a = [[1, 2], [3, 4]]`
- **Tensor:** When we arrange multiple matrices in an array or `Vec`, we get higher rank tensors. This would be beyond our scope in this post and we will keep things simple by restricting ourselves to _2D_ tensors only.

### Matrix Notation and Indexing

When we want to refer to an element inside the matrix, we need a notation to identify specific element.

A Matrix A with m rows and n columns is referred to as an m×n matrix. We denote an individual element within that matrix using subscripts:

$$
A_{i,j}
$$

Where:

- _i_ is the row index (1≤ _i_ ≤_m_)
- _j_ is the column index (1≤ _j_ ≤ _n_)

In code, we usually achieve this by indexing into the array:

```rust
a = [[1, 2], [3, 4]];
println!("{}", a[0][0]); // This would print 1
```

Here is a visual representation of the concept:

$$
\begin{array}{ccc}
\mathbf{Scalar} & \mathbf{Vector} & \mathbf{Matrix} \\
1 & \begin{bmatrix} 1 \\ 2 \end{bmatrix} & \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}
\end{array}
$$


**Note:** Mathematics and programming differ in how we index collection of numbers. Mathematics typically uses 1-based indexing whereas, programming uses 0-based indexing.

### Basic Arithmetic on Matrix
We have defined our Matrix and established its notations. Now let's see how we operate on them.

For tensor of any size, we define the following operations:

#### Element Wise Addition
Element wise addition is only defined for two matrices of the same shape. If A and B are both $m \times n$, then $C=A+B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ + B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\\ 10 & 12 \end{bmatrix} $$


### Element Wise Subtraction
Element wise subtraction is only defined for two matrices of the same shape. If A and B are both $m \times n$, then $C=A-B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ - B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} 5 & 6 \\\ 7 & 8 \end{bmatrix} - \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 4 & 4 \\\ 4 & 4 \end{bmatrix} $$

### Element Wise Multiplication
Element wise multiplication (a.k.a Hadamard Product) is only defined for two matrices of the same shape. If A and B are both $m \times n$, then $C=A \odot B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ \odot B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} 5 & 6 \\\ 7 & 8 \end{bmatrix} \odot \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\\ 21 & 32 \end{bmatrix} $$

Now that we have the mathematical blueprint, let's translate these concepts into Rust code.

## Tensor Implementation
With the math background, now we'll design and implement the `Tensor`. We need a way to store multiple data points and we should be able to index the data structure to access or modify the data inside.

Array matches our requirements and is super fast. However, in Rust arrays can't grow or shrink dynamically at run time. To maintain flexibility, we'll use `Vec` instead. So a basic implementation of our `Tensor` can work well with `Vec<Vec<f32>>`. However, there are two problems in that approach.

1. **Indirection (Pointer Chasing):** `Vec` of `Vec` is very performance intensive operations. Each inner `Vec` is a separate heap allocation. Accessing elements requires jumping to different memory locations. 
2. **Rigidity:** `Vec` of `Vec` would permanently limit our application to a 2D matrix and later, if we want to support higher dimension tensors, we would have to change our code.

To avoid these problems, we'll use two `Vec`s instead. One will hold the data in a flat _1D_ structure and the other will hold the _shape_ definition like this:

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

These two fields should not be accessible directly, we need to define accessors for them and also, we should expose methods for `add`, `sub` and `mul`.

```rust
impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        todo!()
    }

    pub fn data(&self) -> &[f32] {
        todo!()
    }

    pub fn shape(&self) -> &[usize] {
        todo!()
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        todo!()
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        todo!()
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        todo!()
    }
}
```

```rust
pub mod tensor;
```


Now we have defined our data structure and required functions and methods. Let's write a few tests now.

```rust
use build_your_own_nn::tensor::Tensor;

#[cfg(test)]
#[test]
pub fn test_tensor_operations() {
    use std::vec;

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = a.add(&b);
    assert_eq!(c.data(), &[6.0, 8.0, 10.0, 12.0]);
    assert_eq!(c.shape(), &[2, 2]);

    let d = a.sub(&b);
    assert_eq!(d.data(), &[-4.0, -4.0, -4.0, -4.0]);
    assert_eq!(d.shape(), &[2, 2]);

    let e = a.mul(&b);
    assert_eq!(e.data(), &[5.0, 12.0, 21.0, 32.0]);
    assert_eq!(e.shape(), &[2, 2]);
}
```

If we try to run the tests now, it will break. We need to first complete the implementations.

All the implementations till now operate on the data element wise and must match the shape of those two tensors. So, we will add a common method inside the `impl` block and use it to unify all the element wise logic using function pointers. So, the modified `impl` looks like:

```rust
impl Tensor {
    pub fn _element_wise_op(&self, other: &Tensor, op: fn(f32, f32) -> f32) -> Tensor {
        if self.shape != other.shape {
            panic!("Shapes do not match for element-wise operation");
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();

        Tensor::new(data, self.shape.clone())
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        if data.len() != shape.iter().product::<usize>() {
            panic!("Data length does not match shape");
        }
        Tensor { data, shape }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self._element_wise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self._element_wise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self._element_wise_op(other, |a, b| a * b)
    }
}
```

Now, if we run the tests, we can see the tests passing.

```shell
~/git/build-your-own-nn$ cargo test
   Compiling build-your-own-nn v0.1.0 (/home/palash/git/build-your-own-nn)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.30s
     Running unittests src/lib.rs (target/debug/deps/build_your_own_nn-8e7fc48103748a00)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/main.rs (target/debug/deps/build_your_own_nn-fb385501dec7dedb)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/test_tensor.rs (target/debug/deps/test_tensor-25b5f99a2a90f9bb)

running 1 test
test test_tensor_operations ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests build_your_own_nn

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```