### 11/25/24

2 hours - Got data generation working! Randomizing parameters yields acceptable results, not observing any of my previous concerns (bad random patches, audio issues, etc.). ~~~~Setup Slurm scripts for mass data generation.~~ TAL Noisemaker can't be loaded into DawDreamer on Linux! Will just have to generate samples locally on Windows :(.

1 hour - Research methods for audio classification. Promising directions: plain CNN with no activation at the end, generate vector of features directly. Translate audio into spectrogram data. Or, train a convolutional autoencoder, throw out the deocoder, and train a simple mlp. Or, experiement with audio spectrogram transformer.

### 11/26/24

2 hours - Setup model demo script, generate spectrogram, experiment with loss functions, get inputs correct, etc.

### 11/27/24

3 hours

todo:
- research loss functions
- get basic demo working

Initial test looks correct - model doesn't seem to learn anything. This feels a little unsurpring: it may be very hard to derive enough meaningful signal from the spectrogram to predict 88 floats simultaneously. Still, will be interesting to test on more data and a larger model.

To look into:
- Scale up basic CNN with lots of data, see if any better results can be achieved
- Train an autoencoder, throw away the head, and train simple mlp.
- Test audio spectrogram transformer.

### 11/29/24

3 hours

Upping batch size seems to help loss curve, but it gets stuck around 0.085 pretty fast. I suspect the CNN is just not big enough to fully learn

- Get validation metric for checking performance
- Up size of CNN. Resnet?

Even after scaling up to a million parameters, the model doesn't seem to learn anything valuable. Possibly just representation issues. It looks like the model is just learning to predict a number around .50 for each feature, regardless of input value. This would make a lot of sense: the regression just chooses the average because it scores the highest that way. It's not recieving enough granularity in the signal to find any correlations.

### 12/2/24

2 hours

Tested prediction of just one feature, to see if it will learn anything more valuable. And the same thing happens: it learns exactly .50.
This raises another interesting point: not all features are going to have equal levels of predictability: some parameters will not manifest changes very dramatically in the spectrogram.
Work on implementing auto encoder.

### 12/3/24

1.5 hours

Debug autoencoder. Got it working, but loss is all over the place.

### 12/5/24

2 hours

Test autoencoder on simple cifar10 example to verify it works. It appears to be working after 20 epochs. So, I tried to swap out for my data. Got it working, and checking the images and reconstructions it's clear that the decibel representations of the spectrograms are not translating properly to the actual images...rather than using the decibels, I may need the actual pixel values of the spectrogram. Still a little confused by this.

### 12/6/24

1.5 hours

Worked on the spectrogram processing. Got the dataset to load the raw image data, but still seeing crazy loss when training on this. Might need to look into the image format, normalization, etc.

### 12/11/24

3 hours

Debugging loss issue - I just want to get the model to learn to recreate the image well.

Normalizing the image to be between 0 and 1 looks to fix the loss problem: now, the image is gradually being recreated, and loss goes down appropriately. However, it appears there's a memory leak, and torch isn't finding the gpu. Updated torch to cuda version(looks like it accidently reinstalled cpu version). 

For some reason it gets to 30% of the epoch, and gradually slows way down. RAM is going up but gpu memory is staying the same, so there's probably a memory leak issue with my data loading...found that generating images on the fly is just slow, and it's keeping something in memory when it does it. So I just went ahead and generated spectrograms for all of the data. Lesson learned: dynamically creating training data during training is a bad idea.

Autoencoder appears to be working now, I need to use the full size image with a larger model, instead of just a crop. This makes me wonder if the CNN would work better on the normalized data.

### 12/12/24

4 hours

Tried using update image loading/normalization method on the CNN - still having the same issue.

Worked on getting the autoencoder working for the full spectrograms. Ran out of memory on my GPU for 400x400 pixel images, so I need to move to the supercomputer, but I'm going to test with smaller images first. Worked with 128

Now, I need to save the encoder only, and take the latent representation it produces for classification. Added code for this.

### 12/13/24

5 hours

Tested code for classifiers. Initial results show the same behaviour as the CNN: all predictions are about .5. Going to try on a bigger model, just predicting one feature, and look into MSE a bit more.

I'm curious if I just need to let it train for a long time, on a lot of data. It makes sense that the model would initially pick out .5. With enough parameters, couldn't it learn the space in more detail? Letting it cook for a while brought the r2 score up to .14, which is better than before. The predictions look better too.

I trained to just predict one of the parameters. After about 30 min, the r2 score ended up at .99. Great! The actual predictions look excellent as well. If the full multioutput regression fails even after a lot of training, I can still train seperate regression heads for each parameter, and then just combine results at inference time.

Cleaned up my demo/experimental code, created scripts, and moved to the supercomputer for testing.
