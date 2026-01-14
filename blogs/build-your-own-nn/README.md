# Build Your Own Neural Network from Scratch in Rust: From Zero to Image Reconstruction

## Prologue
Machine Learning often felt like a "black box" to me. Every tutorial I found introduced `NumPy` as the baseline requirement. Libraries like `scikit-learn`, `PyTorch`, `TensorFlow`, etc. are excellent for building quick prototypes as well as production-grade models. However, they heavily obscure the underlying mechanics. Hence, I decided to start learning this technology by building it from scratch. 

I have spent years trying to learn Rust. After experimenting with various methods (The Book, RBE, Rustlings, etc.) over the years, I found the missing link: the difficulty lay not in the language, but in the lack of a motivating end-goal.

This project began as a month-long deep dive into Linear Regression. However, my momentum gradually slowed and after numerous iterations, I have finally reached a milestone where I can document this journey. As of today, the project is still evolving.

## Modus Operandi
This guide is designed with a specific philosophy in mind: **Radical Transparency**. We do not start with frameworks or pre-built third-party libraries. We start with a blank file and a single `fn main()`. From there, we will incrementally build the mathematical and structural architecture required to perform complex tasks.

### The Roadmap: From Zero to Image Reconstruction

- **The Blank Canvas:** Initializing the environment and establishing the foundational data structures.
- **The Mathematical Engine:** Implementing tensors, gradients, and backpropagation from scratch.
- **Building the Network:** Constructing layers and activation functions.
- **The Visual Goal:** Training our library to interpret and reconstruct images, proving that 'magic' is just high-dimensional calculus written in a language with high standards.

![Image Reconstruction]()

**Note to the Reader:** This is the roadmap I wish I had two years ago. Whether you are a Rustacean curious about AI or an ML practitioner looking to understand systems-level implementation, this journey is for you.

And that’s where the story begins...

## The Tensor
To build a neural network from scratch, we need to construct the fundamental building blocks first. In the world of Machine Learning, that building block would be a **Tensor**. In simple terms, a tensor is a collection of numbers, organized in a grid.

###  Journey from Scalar to Tensor
To understand the data structure we would be building, we need to develop an intuition first. Let's start building it from scratch as well.

- **Scalar:** We are familiar with this and use it every time we perform arithmetic operations: a single number (e.g. 5). 
    In the world of tensors, we would define them as a tensor of rank 0.
    In programming, this would be a single numeric variable: `x=5`
- **Vector:** When we arrange a collection of numbers, we get a `Vector`.
    In the world of tensors, we would define them as a tensor of rank 1.
    In programming, this would be an array or `Vec` of numeric variables: `a = [1, 2, 3]`
- **Matrix:** When we arrange multiple vectors in an array, we get a matrix.
    In the world of tensors, we would define them as a tensor of rank 2.
    In programming, this would be an array of arrays (or `Vec` of `Vec`s): `a = [[1, 2], [3, 4]]`
- **Tensor:** When we arrange multiple matrices in an array or `Vec`, we get higher rank tensors. This would be beyond our scope in this guide and we will keep things simple by restricting ourselves to _2D_ tensors only.

### Matrix Notation and Indexing

When we want to refer to an element inside the matrix, we need a notation to identify a specific element.

A Matrix A with m rows and n columns is referred to as an m×n matrix. We denote an individual element within that matrix using subscripts:

$$
A_{i,j}
$$

Where:

- _i_ is the row index (1 ≤ _i_ ≤ _m_)
- _j_ is the column index (1 ≤ _j_ ≤ _n_)

In code, we usually achieve this by indexing into the array:

```rust
a = [[1, 2], [3, 4]];
println!("{}", a[0][0]); // Output: 1
```

Here is a visual representation of the concept:

$$
\begin{array}{ccc}
\mathbf{Scalar} & \mathbf{Vector} & \mathbf{Matrix} \\
1 & \begin{bmatrix} 1 \\ 2 \end{bmatrix} & \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}
\end{array}
$$


**Note:** Mathematics and programming differ in how we index a collection of numbers. Mathematics typically uses 1-based indexing, whereas programming uses 0-based indexing.

### Basic Arithmetic on Matrices
We have defined our matrix and established its notation. Now let's see how we operate on them.

For tensors of any size, we define the following operations:

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
Element wise multiplication (a.k.a. _Hadamard Product_) is only defined for two matrices of the same shape. If A and B are both $m \times n$, then $C=A \odot B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ \odot B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} 5 & 6 \\\ 7 & 8 \end{bmatrix} \odot \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\\ 21 & 32 \end{bmatrix} $$

Now that we have the mathematical blueprint, let's translate these concepts into Rust code.

## Tensor Implementation
With the mathematical background, now we'll design and implement the `Tensor`. Let's first kick off the project and then we'll add elements to it. We'll use default `cargo new` for this:

```shell
$ cargo new build_your_own_nn
    Creating binary (application) `build_your_own_nn` package
note: see more `Cargo.toml` keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

```
And that's it. Nothing else. Now let's start putting our thoughts into code.

We need a way to store multiple data points and we should be able to index the data structure to access or modify the data inside.

An array matches our requirements and is super fast. However, in Rust arrays can't grow or shrink dynamically at run time. To maintain flexibility, we'll use `Vec` instead. So a basic implementation of our `Tensor` can work well with `Vec<Vec<f32>>`. However, there are two problems in that approach.

1. **Indirection (Pointer Chasing):** A `Vec` of `Vec`s is a very performance-intensive structure. Each inner `Vec` is a separate heap allocation. Accessing elements requires jumping to different memory locations. 
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
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, TensorError> {
        todo!()
    }

    pub fn data(&self) -> &[f32] {
        todo!()
    }

    pub fn shape(&self) -> &[usize] {
        todo!()
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        todo!()
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        todo!()
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
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
pub fn test_tensor_operations() -> Result<(), TensorError>  {
    use std::vec;

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

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

All the implementations so far operate on the data element wise and must match the shape of those two tensors. So, we will add a common method inside the `impl` block and use it to unify all the element wise logic using function pointers. So, the modified `impl` looks like:

```rust
impl Tensor {
    pub fn _element_wise_op(
        &self,
        other: &Tensor,
        op: fn(f32, f32) -> f32,
    ) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();

        Tensor::new(data, self.shape.clone())
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, TensorError> {
        if shape.len() == 0 || shape.len() > 2 {
            return Err(TensorError::InvalidRank);
        }

        if data.len() != shape.iter().product::<usize>() {
            return Err(TensorError::InconsistentData);
        }
        Ok(Tensor { data, shape })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a * b)
    }
}
```

Now, if we run the tests, we can see the tests passing.

```text
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

**Note:** We will be using standard Rust module system throughout.

Currently the directory structure should look like the following:

```text
src
├── lib.rs
├── main.rs
└── tensor.rs
tests
└── test_tensor.rs
```

## Tensor Display
So far we have written tests for everything to verify operations but we'll need to look at the matrices to see them in a comprehensive and readable format. Looking at the data directly from `Vec` isn't very intuitive.

Let's first try to understand the problem and then we'll fix it. We rewrite the `main` function to inspect the data inside the tensor:

```rust
use build_your_own_nn::tensor::Tensor;
use build_your_own_nn::tensor::TensorError;

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    println!("Tensor data: {:?} {:?}", a.data(), a.shape()); // Output: Tensor data: [1.0, 2.0, 3.0, 4.0] [2, 2]

    Ok(())
}

```

As you can see, the output is a linear array of data. It does not preserve the dimensionality of the tensor. To fix this linear display of tensors and a nice matrix-like format, we'll implement the `Display` trait for our `Tensor` struct, such that any time we want to display the tensor, it will show in a nice formatted way.

The shape `Vec` will help us here. First we define what do the elements map to and here we decide the rules:

1. If the length of `shape` is 1, it is a _vector_ or tensor of rank 1
1. If the length of `shape` is 2, it is a _matrix_ or tensor of rank 2 and the first element of the `shape` vector defines rows and the second element defines columns. By the way, this convention is known as **Row-major**.
1. We don't go beyond _2D_
1. For each row we'll pick out elements matching column length indexing $(row \times cols) + col$

Let's take an example,

$$\begin{bmatrix} 1_{0} & 2_{1} & 3_{2} & 4_{3} \end{bmatrix} \implies \begin{bmatrix} 1_{(0)} & 2_{(1)} \\ 3_{(2)} & 4_{(3)} \end{bmatrix}$$

Let's implement these rules for our tensor now.

First we add the tests as per our desirable matrix look:

```rust
#[test]
fn test_tensor_display_2d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let output = format!("{}", tensor);

    println!("{}", output);

    assert!(output.contains("|  1.0000,   2.0000|"));
    assert!(output.contains("|  3.0000,   4.0000|"));
}

#[test]
fn test_tensor_display_alignment() {
    let tensor = Tensor::new(vec![1.23456, 2.0, 100.1, 0.00001], vec![2, 2]);

    let output = format!("{}", tensor);

    assert!(output.contains("  1.2346"));
    assert!(output.contains("  0.0000"));
}

#[test]
fn test_tensor_display_1d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let output = format!("{}", tensor);
    assert!(output.contains("[1.0, 2.0, 3.0]"));
}
```

And then we implement matching rules to make the tests green.

```rust
impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // As we are dealing with 2D tensors max, we can simply return the debug format for 1D tensors
        if self.shape.len() != 2 {
            return write!(f, "{:?}", self.data);
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        for row in 0..rows {
            write!(f, "  |")?;
            for col in 0..cols {
                let index = row * cols + col;
                write!(f, "{:>8.4}", self.data[index])?;

                if col < cols - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "|")?;
        }
        Ok(())
    }
}
```
Let's rewrite the `main` function and look at the output:

```rust
use build_your_own_nn::tensor::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    )?;

    println!("{}", a);

    println!("{}", b);
    Ok(())
}
```

```text
  |  1.0000,   2.0000|
  |  3.0000,   4.0000|

  |  1.0000,   2.0000,   3.0000|
  |  4.0000,   5.0000,   6.0000|
  |  7.0000,   8.0000,   9.0000|

```

**Challenge to the readers:** I encourage the readers to implement their own formatting. I chose this formatting because I like it, you don't have to stick to this.

## _2D_ Matrix Operations
In the previous operations, we treated matrices like rigid containers—adding or multiplying elements that lived in the exact same "neighborhood." However, to build a neural network, we need to support a few _2D_ operations as well.

The following are a few operations we are going to describe, write tests for and implement in our `Tensor`.

### Transpose
One of the most fundamental transformations in linear algebra involves changing the very orientation of the data. This is known as the **Transpose**. In a transposition operation, the rows of the matrix become columns and the columns become rows.

$$
(A^T​)_{i,j}=A_{j,i}​
$$

Let's take a few examples:

#### Vector Transpose

$$\begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix} \xrightarrow{transpose} \begin{bmatrix} 1 \\ 2 \\ 3 \\ 4 \end{bmatrix}$$

#### Square Matrix Transpose
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \xrightarrow{transpose} \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$$

#### Rectangular Matrix Transpose
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \xrightarrow{transpose} \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}$$

**Note:** In the matrix transpose examples, take a note that the main diagonal elements ($A_{i,j}$ where $i=j$) stay in their positions and don't move. Additionally, in the case of rectangular matrix transposition the shape changes. 

For example, here a $(3 \times 2) \xrightarrow{} (2 \times 3)$ matrix.

### Reduction
A matrix or a vector gives us information about individual elements, but at times we need an aggregation of those individual elements.

Let's look at an example of a matrix which represents sales records of cars in the last three months:

$$
\begin{array}{c|c|c|c}
\mathbf {} & {Maruti} & \mathbf{Hyundai} & \mathbf{Toyota} \\
\hline
Oct  & 1000 & 2000 & 3000 \\
Nov  & 1200 & 1800 & 2000 \\
Dec  & 1500 & 2500 & 2200 \\
\end{array}
$$

This individual representation is great for individual sales of a particular brand in a particular month.

However, if we need to know how many cars were sold in October or how many Maruti cars were sold in the last three months, we need to reduce all the row-wise or column-wise entries into a single number. This operation is known as **Reduction**.

Using reduction we can represent this:

$$
\begin{array}{c|ccc|c}
{} & \mathbf{Maruti} & \mathbf{Hyundai} & \mathbf{Toyota} & \mathbf{Monthly\ Total} \\
\hline
Oct  & 1000 & 2000 & 3000 & 6000 \\
Nov  & 1200 & 1800 & 2000 & 5000 \\
Dec  & 1500 & 2500 & 2200 & 6200 \\
\hline
Brand\ Total  & 3700 & 6300 & 7200 & \\
\end{array}
$$

The 'Brand Total' is a column wise (later represented as Axis 0 sum) reduction and the 'Monthly Total' is a row wise (later represented as Axis 1 sum) reduction.

If we sum across row first and then do another sum of the resultant vector, it will result in the grand sum (the bottom right corner '17200'). This sums up every element in the whole matrix into a single scalar value.

$$
\begin{array}{c|ccc|c}
\mathbf {} & {Maruti} & \mathbf{Hyundai} & \mathbf{Toyota} & \mathbf{Monthly\ Total} \\
\hline
Oct  & 1000 & 2000 & 3000 & 6000 \\
Nov  & 1200 & 1800 & 2000 & 5000 \\
Dec  & 1500 & 2500 & 2200 & 6200 \\
\hline
\mathbf{Brand\ Total}  & 3700 & 6300 & 7200 & \mathbf{17200} \\
\end{array}
$$

### Dot Product
We have already seen how to multiply two matrices or vectors element-wise. However, there is another multiplication operation we can perform, known as the **Dot Product**. It is slightly more involved, as it combines element-wise multiplication and a reduction operation into a single step.

The dot product of two matrices A and B of length n is defined as:
$$
A \cdot B = \sum_{i=1}^{n} A_i B_i
$$

Let's take a few examples.

#### Vector Vector Dot Product
Here is an example of a dot product between two vectors:

$$
\begin{bmatrix}
\color{red}1 \\ \color{blue}2 \\ \color{green}3 \\ \color{yellow}4
\end{bmatrix} 
 
\cdot 
 
\begin{bmatrix} 
\color{red}1 \\ \color{blue}2 \\ \color{green}3 \\ \color{yellow}4
\end{bmatrix}
 
=

\color{red}{(1 \times 1)} \color{white}+ \color{blue}{(2 \times 2)} \color{white}+ \color{green}{(3 \times 3)} \color{white}+ \color{yellow}{(4 \times 4)}

\color{white}=30
$$

#### Matrix Vector Dot Product
In a Matrix-Vector dot product, we calculate the dot product of every row from the matrix with the single the column of the vector.

To perform a dot product between a matrix $A$ and a vector $v$, the number of columns in the matrix must equal the number of elements (rows) in the vector.

If matrix $A$ has the shape $(m \times n)$ and vector $v$ has the shape $(n \times 1)$, the resulting vector w will have the shape $(m \times 1)$.

Matrix Vector dot product is defined as:
$$
C_{m,1} = A_{m, n}v_{n. 1}
$$

Let's take an example:

$$

\begin{bmatrix}
\color{red}{1} & \color{red}{2} & \color{red}{3} \\ 
\color{yellow}{4} & \color{yellow}{5} & \color{yellow}{6}
\end{bmatrix} 
\cdot
\begin{bmatrix}
\color{cyan}{7} \\
\color{cyan}{8} \\
\color{cyan}{9}
\end{bmatrix} 
=
\begin{bmatrix}
\color{red}{[1, 2, 3]} \cdot \color{cyan}{[7, 8, 9]} \\
\color{yellow}{[4, 5, 6]} \cdot \color{cyan}{[7, 8, 9]}
\end{bmatrix}
=
\begin{bmatrix}
(\color{red}{1} \times \color{cyan}{7} + \color{red}{2} \times \color{cyan}{8} + \color{red}{3} \times \color{cyan}{9}) \\
(\color{yellow}{4} \times \color{cyan}{7} + \color{yellow}{5} \times \color{cyan}{8} + \color{yellow}{6} \times \color{cyan}{9})
\end{bmatrix}
=
\begin{bmatrix}
50 \\
122
\end{bmatrix}
$$

#### Matrix Matrix Dot Product
In a Matrix-Matrix dot product (often simply called **Matrix Multiplication**), we don't just multiply corresponding "neighborhoods." Instead, we calculate the dot product of every row from the first matrix with every column of the second matrix.

To perform a dot product between matrix $A$ and matrix $B$, the number of columns in $A$ must equal the number of rows in $B$.

If $A$ is $(m \times n)$ and $B$ is $(n \times p)$, the resulting matrix $C$ will have the shape $(m \times p)$.

Matrix Multiplication is defined as:
$$
C_{m,p} = A_{m, n}B_{n. p}
$$

A simple way to think about matrix multiplication is to think of dot product of $A$ and $B^T$.

Let's take an example:

$$
\begin{bmatrix}
\color{red}1 & \color{red}2 & \color{red}3 \\ 
\color{yellow}4 & \color{yellow}5 & \color{yellow}6
\end{bmatrix} 

\cdot

\begin{bmatrix}
\color{cyan}7 & \color{magenta}8 \\
\color{cyan}9 & \color{magenta}10 \\
\color{cyan}11 & \color{magenta}12
\end{bmatrix} 

=

\begin{bmatrix}
\color{red}{[1, 2, 3]} \cdot \color{cyan}{[7, 9, 11]} & \color{red}{[1, 2, 3]}\cdot \color{magenta}{[8, 10, 12]} \\
\color{yellow}[4, 5, 6] \cdot \color{cyan}{[7, 9, 11]} & \color{yellow}[4, 5, 6] \cdot \color{magenta}{[8, 10, 12]} \\
\end{bmatrix}
=
\begin{bmatrix}
(\color{red}{1} \times \color{cyan}{7} + \color{red}{2} \times \color{cyan}{9} + \color{red}{3} \times \color{cyan}{11}) & 
(\color{red}{1} \times \color{magenta}{8} + \color{red}{2} \times \color{magenta}{10} + \color{red}{3} \times \color{magenta}{12}) \\
(\color{yellow}{4} \times \color{cyan}{7} + \color{yellow}{5} \times \color{cyan}{9} + \color{yellow}{6} \times \color{cyan}{11}) & 
(\color{yellow}{4} \times \color{magenta}{8} + \color{yellow}{5} \times \color{magenta}{10} + \color{yellow}{6} \times \color{magenta}{12})
\end{bmatrix}
=
\begin{bmatrix}
58 & 64 \\
139 & 154
\end{bmatrix}
$$