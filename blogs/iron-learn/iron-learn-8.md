# Proving the Universal Approximation Theorem with Rust

Once I was done with the CUDA Integration in Python, it was time to return to the Rust program. I had to replicate the Neural Network into my Rust program.

I took a look at the code base, all the code was dumped into one single file, violating my personal coding hygiene of 'no more than 500 lines in a file, until absolutely necessary'. Apart from that, the way the logistic regression program was written in raw CUDA launches without following any structure made me concerned that I would have to write duplicate modules for Neural Network to support both CPU and GPU.

I needed a plan to make things unified at the `Tensor` level, beyond that, things can differ but any high level implementation should call `Tensor` modules and things should work without worrying about where the math is being performed.

## The Plan
1. Make a `TensorFamily` enum to determine where the mathematics would execute
1. Write a  `TensorBackend` trait which holds all the methods of `Tensor`
1. Implement the  `TensorBackend` trait for `CpuTensor` and `GpuTensor` structs
1. Finally, rewrite the `Tensor` struct which works as the `Factory`

While doing so, I stumbled upon some new learning in Rust - the dynamic trait and another rabbit hole...

## The Pivot
The plan looked simple on paper. However, it did not work in the real code. The compiler came back with multiple errors. I tried to use `dyn` trait for `TensorBackend`. I tried to resolve a few. Some I fixed, I understood a few new concepts and why Rust prevented me from using a recursive memory allocation pattern and I got stuck. Compiler was very reasonable but I was being completely unreasonable in my plan.

After around two hours of fighting with the compiler, I again had another thought of shutting down the project. I questioned my choices and left the desk for a walk around the block.

There came the solution, I don't need to make the actual `Tensor` unified. No matter what I did, I would still need to make two different execution methods, one for CPU and another for GPU. The user of the library (ironically, that is just me), would make the choice of using GPU or CPU on their work load. They may have installed high end GPUs but for a simple XOR operation Neural Network test, the GPU will actually be slower. I should not make assumptions and must leave the choice to the user.

With this newfound reconciliation, I returned to my desk. I devised a new design altogether, where a consistent set of methods will be exposed for both hardware types. Only difference would be: the CPU-bound tensor can query the memory immediately and return result  while the GPU-based tensor needs an explicit D2H data copy mechanism. Until the D2H call happens, all data resides on GPU memory.

With this new idea, I abandoned all my plans and started fresh with writing GPU-based program separately.

However, this also did not go well, after a few more rounds of errors I stopped GPU programming completely.

It was really a devastating and deeply demoralizing moment for me. My dream was shattering in front of my eyes. I knew there was no way a heavy workload would be completed by my CPU. I need to work on the GPU side. But something in my mind told me quietly, 'don't worry, you will do it, but just not right now'. Somehow, I followed my inner voice and kept aside my thinking brain for few hours. I wrote the Rust CPU-bound neural network, following the Python script.

```rust
/// Element-wise sigmoid activation.
pub fn sigmoid<T>(input: &T) -> Result<T, String>
where
    T: CpuTensor<f32>,
{
    input.sigmoid()
}

/// Derivative of sigmoid; expects the activation output as input.
pub fn sigmoid_prime<T>(output: &T) -> Result<T, String>
where
    T: CpuTensor<f32>,
{
    let one_minus_out = T::ones(&output.get_shape()).sub(output)?;
    let res = output.multiply(&one_minus_out);

    res
}

/// Element-wise hyperbolic tangent activation.
pub fn tanh<T>(input: &T) -> Result<T, String>
where
    T: CpuTensor<f32>,
{
    input.tanh()
}

/// Derivative of `tanh`, expects activation output as input.
pub fn tanh_prime<T>(output: &T) -> Result<T, String>
where
    T: CpuTensor<f32>,
{
    let out_squared = output.multiply(output)?;

    let ones = T::ones(&output.get_shape());

    ones.sub(&out_squared)
}

/// Trait describing a loss function used for training and backpropagation.
///
/// Implementors must provide methods to compute the scalar loss tensor and
/// the derivative (loss prime) used as the starting point for backprop.
pub trait LossFunction<T>
where
    T: CpuTensor<f32>,
{
    /// Calculates the loss value (used for reporting).
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String>;

    /// Calculates the derivative of the loss w.r.t the predicted output (used for backpropagation).
    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String>;
}

/// Mean squared error loss implementation.
pub struct MeanSquaredErrorLoss;

impl<T> LossFunction<T> for MeanSquaredErrorLoss
where
    T: CpuTensor<f32>,
{
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String> {
        let error_diff = predicted.sub(actual).unwrap();
        let sq_err = error_diff.multiply(&error_diff).unwrap();

        let length = sq_err.get_shape().iter().product();

        sq_err.sum().unwrap().scale(1.0 / (length as f32))
    }

    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String> {
        let n = actual.get_shape().iter().product();
        let factor = 2.0 / (n as f32);

        predicted.sub(actual).unwrap().scale(factor)
    }
}

/// Fully-connected linear layer holding weights and an optional input cache.
pub struct LinearLayer<T>
where
   T: CpuTensor<f32>,
{
    weights: T,
    input_cache: Option<T>,
    name: String,
    layer_type: LayerType,
}

impl<T> LinearLayer<T>
where
    T: CpuTensor<f32>,
{
    fn _initialize_weights(
        input_size: u32,
        output_size: u32,
    ) -> Vec<f32> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let w_data: Vec<f32> = (0..(input_size * output_size))
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        w_data
    }
    pub fn new(
        input_size: u32,
        output_size: u32,
        name: &str,
    ) -> Result<Self, String> {
        let w_data = Self::_initialize_weights(input_size, output_size);

        let weights = T::new(vec![input_size, output_size], w_data).unwrap();

        Ok(Self {
            weights,
            input_cache: None,
            name: name.to_string(),
            layer_type: LayerType::Linear,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        self.input_cache = Some(input.add(&T::zeroes(input.get_shape()))?);
        let matmul = input.mul(&self.weights)?;
        Ok(matmul)
    }

    fn backward(&mut self, output_error: &T, lr: f32) -> Result<T, String> {
        let input = self.input_cache.as_ref().ok_or("No forward pass cache!")?;

        // Calculate Input Error: error * weights.T
        let w_t = self.weights.t()?;
        let input_error = output_error.mul(&w_t)?;

        // Calculate Weights Gradient: input.T * error
        let input_t = input.t()?;

        let weights_grad = input_t.mul(output_error)?;

        // Update Parameters
        let w_step = weights_grad.scale(lr)?;
        self.weights = self.weights.sub(&w_step)?;

        Ok(input_error)
    }
}

/// Activation wrapper layer that applies element-wise activation functions.
pub struct ActivationLayer<T>
where
    T: CpuTensor<f32>,
{
    layer_type: LayerType,
    output_cache: Option<T>,
    name: String,
}

impl<T> ActivationLayer<T>
where
    T: CpuTensor<f32>,
{
    /// New takes two function pointers as input
    pub fn new(name: &str, layer_type: LayerType) -> Self {
        Self {
            output_cache: None,
            name: name.to_string(),
            layer_type,
        }
    }
}

impl<T> Layer<T> for ActivationLayer<T>
where
     T: CpuTensor<f32>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        let (activation, _) = get_activations(&self.layer_type);
        // Call the passed-in activation function
        let output = (activation)(input)?;

        // Caching the output for the backward pass
        self.output_cache = Some(output.add(&T::zeroes(output.get_shape()))?);
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, _lr: f32) -> Result<T, String> {
        let out = self
            .output_cache
            .as_ref()
            .ok_or_else(|| "No output cache found for backward pass".to_string())?;

        let (_, activation_prime) = get_activations(&self.layer_type);

        // Call the passed-in activation prime function
        // Note: Many derivatives (like sigmoid/tanh) use the output 'y' rather than input 'x'
        let prime = (activation_prime)(out)?;

        prime.multiply(output_error)
    }

    fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }
}

pub struct NeuralNet<T>
where
    T: CpuTensor<f32>,
{
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub loss_fn: Box<dyn LossFunction<T>>,
}

impl<T> NeuralNet<T>
where
    T: CpuTensor<f32>,
{
    pub fn new(
        layers: Vec<Box<dyn Layer<T>>>,
        loss_fn: Box<dyn LossFunction<T>>,
    ) -> Self {
        Self {
            layers,
            loss_fn,
        }
    }
    /// Append a layer to the network.
    ///
    /// The provided box must implement the `Layer` trait for the network's
    /// tensor type `T`.
    pub fn add(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    /// Run a forward pass and return the network output for `input`.
    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::zeroes(input.get_shape())).unwrap();

        for layer in &mut self.layers {
            output = layer.forward(&output).unwrap();
        }
        Ok(output)
    }

    pub fn fit<F>(
        &mut self,
        x_train: &T,
        y_train: &T,
        epochs: usize,
        base_lr: f32,
    ) -> Result<(), String>
    where
        F: FnMut(usize, f32, f32, &mut Self),
    {
        let lr_min = 1e-6;

        for i in 0..epochs {

            print!("\rProcessing epoch: {}/{epochs}", i);
            io::stdout().flush().unwrap();

            let mut output = x_train.add(&T::zeroes(x_train.get_shape())).unwrap();
            for layer in &mut self.layers {
                output = layer.forward(&output).unwrap();
            }
            T::synchronize(); // no-op for CPU but needed here to maintain consistency with GPU

            let err = self.loss_fn.loss(y_train, &output);
            T::synchronize();

            let mut error_prime = self.loss_fn.loss_prime(y_train, &output).unwrap();

            for layer in self.layers.iter_mut().rev() {
                error_prime = layer.backward(&error_prime, base_lr).unwrap();
            }
            T::synchronize();
        }

        T::synchronize();
        Ok(())
    }
}
```

After around three hours, I was able to finally run my first Rust Neural Network program. Things went pretty well. To be honest, I never expected it to go so smooth. I ran the XOR test in both Python script and Rust program. They showed not exact but very similar results. It was not the exact same result because, the weight initialization followed random sequence without the same seed.

Another success, another idea, another play time...

Approved!!!

## The Universal Approximation Theorem
The Universal Approximation Theorem states that, "Given at least one hidden layer in a neural network and enough time neural networks can approximate any continuous function".

Well, now I have a neural network running and it passed the non-linearity test with XOR operation. I can leave the computer switched on overnight to approximate any function. So, why not try it?

I needed a function to be approximated. I became a little dramatic here.

I wrote a few chits writing `1` to `10`, `+`, `-`, `*`, `/`, `^` and put them in a bowl and picked up 15 times. Please don't judge me. I still have no answer why I did that.
![Math Chits](${iron-learn-8-math-chits} "Math Chits")

Anyways, I came up with this equation:

a(x) = (x² + x³ + x⁴ + x⁶) / x⁵

And this gave me this plot

![Function Plot](${iron-learn-8-actual-plot} "Actual Plot")

I made an arbitrary rule, the blue points are `true` and red points are `false`. I sampled 25 points for training and 6 for testing and tested my neural network against it.

 First attempt went unsuccessful. I could not find the reason. I tried it with the Python script. That also failed. At that point, I had doubt if UAT actually can be applied to this function.
 
 I needed an answer. To solve the tension between my program output and UAT, I wrote another `sklearn` script to run against the same data. This time it went successful, proving my programs were incorrect.

 Here comes the debugging hat...

A few `print()` and `println!()` statements later, I found the issue. The culprit surprisingly was Normalization. It helped me in all earlier cases but this time, it failed me.

 I was normalizing the input and denormalizing the output, which made the prediction result incorrect. I removed the normalization and denormalization layer and it started working.

 Lesson learnt, not all the time you would need normalization and denormalization.

Voila!!!

```shell

╔════════════════════════════════╗
║ Iron Learn v5
║ Mode: CPU
║ Learning Rate: 0.0001
║ Epochs: 100
║ Data Path: data.json
╚════════════════════════════════╝

Predicted: 0.0000, Actual: 0.0000, ✓
Predicted: 0.9999, Actual: 1.0000, ✓
Predicted: 1.0000, Actual: 1.0000, ✓
Predicted: 0.0000, Actual: 0.0000, ✓
Predicted: 0.0024, Actual: 0.0000, ✓
Predicted: 0.0000, Actual: 0.0000, ✓

```

The output confirms two things:
1. Universal Approximation Theorem holds true
1. My Neural Network implementation is correct


## Inventory Check
Another day just passed by fixing things. Not only in code but also in my mind. The day played heavy with my emotions.

Anyways, at that moment I had all these in the inventory:
1. A CPU Linear regression
2. A CPU Logistic Regression
3. A GPU Linear Regression
4. A GPU Logistic Regression
5. A Python script to train GPU powered neural network
6. A Rust program to train Neural Network using CPU

At that point, the math was ready, the code was ready, the network proved to be working. I was ready to do something more with it, and I did. Something unexpected happened and I stumbled upon some great knowledge chunks.
