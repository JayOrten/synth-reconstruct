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