### 11/25/24

2 hours - Got data generation working! Randomizing parameters yields acceptable results, not observing any of my previous concerns (bad random patches, audio issues, etc.). ~~~~Setup Slurm scripts for mass data generation.~~ TAL Noisemaker can't be loaded into DawDreamer on Linux! Will just have to generate samples locally on Windows :(.

1 hour - Research methods for audio classification. Promising directions: plain CNN with no activation at the end, generate vector of features directly. Translate audio into spectrogram data. Or, train a convolutional autoencoder, throw out the deocoder, and train a simple mlp. Or, experiement with audio spectrogram transformer.

### 11/26/24

2 hours - Setup model demo script, generate spectrogram, experiment with loss functions, get inputs correct, etc.