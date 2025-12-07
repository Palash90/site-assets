I started writing a python program which can reconstruct a black and white image by looking at the data.

I came across several problems while doing this.

The image was large 474 * 474 pixels. It was taking long time to train.

If I have to run the program, it starts from epoch zero, adding to the pain.

I thought of adding a load and save weights and biases. So that, in the event of a machine crash, I can resume the learning from the last saved checkpoint data.

Then,I ran it for 10000 turns, then again, from there another 30000 runs. It did not go well.

Then I tried resizing the image to 200 * 200 pixels. It started giving some result. So, I changed the learning rate as well as the epochs and tried to find if it can be supported.

On the way I thought, what if I can build row by row of the image. What if I can change the learning rate on the fly.

I did some research on my thoughts and found about Stochastic Gradient Descent, Lerning rate decay. I applied the cos learning decay function and it smoothly started showing correct result.

After running for almost 1.2 million iterations, could see generated output very close to input. The input was a very complex high dimensional function, far beyond simple XOR gate test set or even the logistic regression dataset. This proves that my neural network was exhibiting properties of Universal Approximation Theorem. I can now use this network to do bigger things.

But before that, out of curiosity, I saved the weights and biases and tried to test it against different blank canvas sizes like 512, 50, 1024 etc. In every blank canvas it drew the image.

So, I basically made a very high complexity, inefficient, uber expensive image scaler.


Time for something bigger.