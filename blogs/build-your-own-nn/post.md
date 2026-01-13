# Build Your Own Neural Network from Scratch in Rust: From Zero to Image Reconstruction

## Prologue
Machine Learning often felt like a "black box" to me. Every time I started learning it, I was introduced to `Numpy` at the very least. Libraries like `scikit-learn`, `PyTorch`, `TensorFlow`, etc. are excellent for building quick prototypes as well as production-grade models. But they heavily obscure the underlying mechanics.

Hence, I decided to start learning this technology by building it from scratch. After experimenting with various implementations over the years, I found the missing link: the difficulty lay not in the language, but in the lack of a motivating end-goal.

This project began as a month-long deep dive into Linear Regression. However, the pace faded away gradually and after numerous iterations, I have finally reached a milestone where I can document this journey. As of this writing, I am still building on it.

## Modus Operandi
This series is designed with a specific philosophy in mind: **Radical Transparency**. We do not start with frameworks or pre-built libraries (we do not even install any of them). We start with a blank file and a single `fn main()`. From there, we will incrementally build the mathematical and structural architecture required to perform complex tasks.

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

- **Scalar:** We are familiar with this and use it every time we perform arithmetic operation: a single number (e.g. 5). 
    In the world of tensors, we would define them as tensor of order 0.
    In programming, this would be a single numeric variable: `x=5`
- **Vector:** When we arrange a collection of numbers, we get a `Vector`.
    In the world of tensors, we would define them as tensor or order 1.
    In programming, this would be an array or `Vec` of numeric variables: `a = [1, 2, 3]`
- **Matrix:** When we arrange more than one vectors in an array, we get matrix.
    In the world of tensors, we would define them as tensor of order 2.
    In programming, this would be an array of arrays (or `Vec` of `Vec`s): `a = [[1, 2], [3, 4]]`
- **Tensor:** When we arrange more than one matrices in an array or `Vec`, we get higher order tensors. This would be beyond our scope in this post and we will keep things simple by restricting ourselves to _2D_ tensors only.

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


**Note:** Mathematics and programming differ in how we index collection of numbers. Mathematics prefers to use 1-based index but programming uses 0-based index.

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

We now have enough mathematical background to start with the first set of implementations.