# Making a Binary Classifier
Once I made the linear regression program to work correctly on a real dataset, my confidence was boosted. I then thought of the next level. The easiest one to cover was to integrate a sigmoid function to wrap around the linear equation and get the binary classifier working.

The data was easy to find. I searched for few. Many of them was having string input as one of the columns. I had to discard them as my program relies on numeric data and I had no encoder or decoder ready yet. After skipping a few, I finally stuck with the student's Pass/Fail dataset. It's a nice minimal dataset of 100 data rows with two input columns and one output column.

## The First Run
As earlier, I again broke the dataset in 90:10 ratio for train and test. I implemented the `sigmoid` function on my linear regression algorithm and ran. Thankfully, I did not end up with `NaN` this time.

```rust
fn sigmoid(z: Tensor<f64>) -> Tensor<f64> {
    let result = Tensor::exp(&-z);
    let shape = result.get_shape();
    let result = result.get_data().iter().map(|t| 1.0 / (1.0 + t)).collect();

    Tensor::new(shape, result).unwrap()
}
```

However, the program returned a 60% accuracy rate on 10 test examples.

## The Cross Validation
Like last time, I again wrote a sklern program to validate my results. The python program was giving 100% accuracy. I again started looking around the same steps I performed last time. First checked normalization. Well, it was not there. So, I fixed it the first. Then checked learning rate and iterations. They were also set on wrong numbers.

After fixing all these, I checked again and this time, I hit 100% accuracy rate.

# Next steps, more varied data

Total 3921 rows, 22 columns each - One is the spam/ham output, one with row number and rest 20 are data

Took 391 for test
Rest for Train
The data was with yes/no and big/small/none etc. I took them up and changed them to different numbers. 

Removed the date time column.

Finally it was a 19 column data set for 3530 training rows and 391 test rows.

Copied the data in my expected input format.

For the first time, I was running data so huge on my machine for my poor performance program. The linear regression program takes around 40 seconds for 321 rows of 6 columns each. This is 10.5 times more rows with 3 times more columns. And it is running a school grade matrix multiplication algorithm. To better estimate how much time, it will take to finish, I put a `println!` statement for every 100 iterations to measure, how much it would take to finish 5000 iterations. It was taking beyond my patience can control. I switched to a smaller log of 10 iterations. 10 iterations are taking around 35 seconds now. So, 5000 iterations will take around 17500 seconds, which is around 5 hours. I discarded the plan and I changed number of training loops to 1000, such that it would wrap up in one hour.

Finally waiting for approximately an hour, my program returned some result for me to show.

Well, it has done a decent job.

Results:
Total samples: 391
Correct predictions: 362
Accuracy: 92.58%

I ran the same for in my sklearn based program and it returned almost identical result, just in a few seconds...
Results:
Total samples: 391
Correct predictions: 364
Accuracy: 93.09%

The next attempt was extreme. So, I decided to actually run the program for 5000 iterations and see if it goes closer to the sklearn program or not. I made the changes in the program, ran it and went for bed that day.

I saw my machine crashing just after about an hour of starting the program with 5000 iterations. I stopped it, switched off my laptop and went back to bed.

My intention was never let it run fast and with 100% accuracy in the first go. I just wanted to learn how things works. So, I am happy with the result my program gave me.



