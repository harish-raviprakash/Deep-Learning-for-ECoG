# Deep-Learning-for-ECoG
Official implementation for [Deep Learning Provides Exceptional Accuracy to ECoG-Based Functional Language Mapping for Epilepsy Surgery](
https://www.frontiersin.org/articles/10.3389/fnins.2020.00409/full#:~:text=10.3389%2Ffnins.2020.00409-,Deep%20Learning%20Provides%20Exceptional%20Accuracy%20to%20ECoG%2DBased,Language%20Mapping%20for%20Epilepsy%20Surgery&text=The%20success%20of%20surgical%20resection,regions%2C%20while%20removing%20pathological%20tissues.)

Please cite our paper if you use this code.
RaviPrakash, Harish, et al. "Deep Learning provides exceptional accuracy to ECoG-based Functional Language Mapping for epilepsy surgery." Frontiers in Neuroscience 14 (2020): 409.


__How to interpret matlab codes__

Data is assumed to be of dimensions M×N with M = the number of channels and N = number of datapoints.
The paradigm used is 30seconds control and 30seconds active task. 

Step 1: Run extractData.m

This will load each subject’s data (Please provide the path) and call createSlidingActiveTime.m

createSlidingActiveTime.m will extract the AR, PSD, Peak-to-Peak and Mean, skew, kurtosis and Hjorth features from the active blocks of data.

Step 2: Run getFeatureLabels.m

This will assign block labels for each block in the paradigm, where a block is 30 seconds of data [control/active task].

__How to interpret python codes__

train.py includes code to train the model and plot the training curves.

loadData.py includes functions to read the data and reshape them for the deep learning architectures.

customLayers2.py includes functions from [Keras](https://faroit.com/keras-docs/2.1.5/layers/recurrent/)

There are several different network architectures in modelFiles2.py. Basic architecture:

|     Layer                                |     Filter   Size                  |
|------------------------------------------|------------------------------------|
|     1D Convolution (Time Domain RNN)     |     128, 64,   128                 |
|     Dense (Frequency Domain Features)    |     64                             |
|     LSTM (Time Domain RNN)               |     8                              |
|     Dense (Domain Fusion Network)        |     64, 32, 2                      |
|     Global Hyperparameters               |     Value                          |
|     Loss                                 |     Categorical   Cross-Entropy    |
|     Optimizer                            |     Adam                           |
|     Learning Rate                        |     0.001                          |
|     Exponential decay rate 1             |     0.9                            |
|     Exponential decay rate 2             |     0.999                          |
|     Decay every ‘n’ epochs               |     25                             |

![Basic Architectures](https://github.com/harish-raviprakash/Deep-Learning-for-ECoG/blob/master/at_ar.JPG)

__Running the python code__

Please fill in the default paths under int_main of rtfmPSD_P2P_all.py and run python rtfmPSD_P2P_all.py

or during the call via ‘python rtfm_psdP2P_all.py --data_root_dir /path/to/time-series/data --file1  /path/to/psd/data --file2 /path/to/labels –-timeFeats [‘’,‘’] –-save_dir /path/to/store’
