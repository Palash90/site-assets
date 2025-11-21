# Making a Binary Classifier
Once I got the linear regression program working on a real dataset, my confidence was boosted. I immediately looked forward. The easiest next step was to integrate a sigmoid function to wrap around the linear equation and get a binary classifier working.

The data was easy to find. I searched for a few options, but many of them had string input as one or more columns. I had to discard them as my program relies on numeric data and I had no encoder or decoder ready yet. After skipping a few, I finally stuck with the "Student Pass/Fail" dataset. It's a nice minimal dataset of 100 data rows with two input columns and one output column.

## The Math Behind
Linear regression models try to learn the linear relation between the input and output. Logistic regression on the other hand squeezes the line within two probability values - 0 and 1.

### The Sigmoid Function
The linear model calculates - $z = \mathbf{w}^T X + b$. To turn this value into a probability, we use **Sigmoid function**

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

![Sigmoid Function](https://palash90.github.io/site-assets/blogs/iron-learn/iron-learn-2-logistic-fn.png "Sigmoid Function")

This function takes any real number input ($z$) and maps it to a value $\hat{y}$ between 0 and 1. If $\hat{y}$ is above 0.5, we classify the input as a "1" (Pass/Spam); otherwise, it's a "0" (Fail/Ham).

### The Loss Function
For linear regression we used MSE earlier. However, MSE is not ideal for logistic regression. The standard choice is Cross-Entropy Loss, whose gradient simplifies to the difference between prediction and actual label. This subtraction looks similar to MSE but is not mathematically identical.

I could have used the sophisticated algorithms to help me with replacing loss function but I kept it at the minimum and simply calculated the gradient update by subtracting the prediction output from the actual result.

## The First Run
As before, I broke the dataset in 90:10 ratio for train and test. I implemented the `sigmoid` function on top of my linear regression algorithm and ran it. Thankfully, I did not end up with `NaN` this time.

```rust
fn sigmoid(z: Tensor<f64>) -> Tensor<f64> {
    let result = Tensor::exp(&-z);
    let shape = result.get_shape();
    let result = result.get_data().iter().map(|t| 1.0 / (1.0 + t)).collect();

    Tensor::new(shape, result).unwrap()
}

pub fn gradient_descent(
    x: &Tensor<f64>,
    y: &Tensor<f64>,
    w: &Tensor<f64>,
    l: f64,
    logistic: bool,
) -> Tensor<f64> {
    let data_size = *(x.get_shape().first().unwrap()) as f64;
    let lines = x.mul(w).unwrap();

    let prediction = match logistic {
        true => sigmoid(lines),
        false => lines,
    };

    let loss = prediction.sub(&y).unwrap();
    let d = x
        .t()
        .unwrap()
        .mul(&loss) // Multiply `X` to loss
        .unwrap()
        //.add(&(w.scale(lambda/data_size)))// Add Regularization to parameters
        //.unwrap()
        .scale(l / data_size); // Scale gradient by learning rate

    w.sub(&d).unwrap()
}
```

However, the program returned a 60% accuracy rate on 10 test examples.

## The Cross Validation
Like last time, I again wrote a sklearn program to validate my results. The Python program was giving 100% accuracy. I started looking around at the same steps I performed last time. First, I checked normalization. Well, it wasn't there. So, I fixed that first. Then I checked learning rate and iterations. The numbers were off.

After fixing all these, I ran it again. This time, I hit 100% accuracy rate.

## Next steps: More varied data
I was happy that my program matched accuracy to that of the sklearn's. However, I wondered if my program would break with a larger dataset. An obvious limitation was on the hardware and the program itself. It limits me to smaller datasets. However, I would be happy to spend some time to see my program return acceptable result. 

### The dataset
I searched for another valid and preferably bigger dataset. After checking a few, I finally settled on an email spam dataset. It has a total of 3921 rows, 22 columns each: one spam/ham output, one row number and rest are numeric, text and date-time data.

### Preprocessing and Train-Test Split
Few columns in the dataset came with `yes/no` and `big/small/none` text. I converted all the text data into numeric values; `yes/no` became `0/1` and `big/small/none` became `2/1/0`, and so on. 

I noticed one column with datetime information. I gave it a thought - it can help the algorithm decide on spam classification but I felt lazy to convert them in a meaningful way. I dropped the idea.

Finally, the data was cleaned up and converted to pure numbers.

Then comes the train-test split. I had 3921 records, 19 pre-processed columns in each. I kept 391 records separate for testing and the rest for Training. The data came in CSV but my program doesn't yet support CSV, so I had to convert the data into my expected `json` format.

## The Next Run
For the first time, I was feeding a huge amount of data into my poor-performance program. The linear regression program takes around 40 seconds for 321 rows of 6 columns. This new dataset had 10.5 times more rows and 3 times more columns. And remember, it is running a basic matrix multiplication algorithm in a sequential fashion.

To estimate how much time, it would take for 5000 iterations, I put a `println!` statement for every 100 iterations. It was taking longer than my patience could handle. I switched to a smaller log of 10 iterations. 

10 iterations were taking around 35 seconds. That implies 5,000 iterations will take around 17,500 seconds, roughly 5 hours. I discarded the plan and I changed number of training loops to 1000, hoping to wrap up within an hour.

And the wait began...

Finally, after approximately an hour, my program returned some results for me to analyze. It performed a decent job.

**Results:**
- Total samples: 391
- Correct predictions: 362
- Accuracy: 92.58%

I ran the same dataset in my sklearn-based program and it returned almost identical results just in a matter of seconds -
**sklearn results:**
- Total samples: 391
- Correct predictions: 364
- Accuracy: 93.09%

## The Aftermath
I was happy that with only 1000 iterations, my program returned result close to a time tested production grade library with all possible optimizations. 

However, I didn't stop testing. The next attempt was extreme. I decided to actually run the program for the full 5000 iterations to see if it can match a closer accuracy to that of the sklearn program. I made the changes, hit run and went to bed.

One hour later...

My machine crashed just after about an hour of starting the program with all 5000 iterations. I stopped it, switched off my laptop and went back to sleep.

My intention was never to make it run fast and with 100% accuracy on the first go. I just wanted to learn how the "magic" works. In that regard, I am very happy with the result.

