# Deep Back Projection

Joint work with Max Ruby. Purdue University course project for BME 595.  

## Intro
The topic of this project is to take a look at a problem in medical imaging reconstruction.
In this kind of problem we desire to reconstruct an image from projection data of some
kind. For instance, we may be given an x-ray of a skull, and our objective would be to
mathematically reconstruct the skull itself. Obviously, accuracy in the reconstruction is of
intrinsic importance. Medical professionals rely on such images to make informed decisions
about medical care.  
In this project, we attempt to use a neural network to accurately carry out this task.  

## Prior Work
Historically, the Radon transform (and its inverse) has been used to perform this task. The
Radon transform is simply a collection of the projection data of an image. Armed with
the Radon transform (graphically called a sinogram) of the image, we can apply the inverse
Radon transform through a technique called filtered back projection in order to reconstruct
the image.  
For various reasons, sometimes it is often infeasible in medical imaging to generate a full
sinogram of the image with projection data at every angle. When this is the case, sparse-view filtered back projection is sometimes used. A sparse-view filtered back projection has the a much lower number of views taken of the object. When a lower number of views is taken
there is less projection data, which is to say that the sinogram contains less information.
The results of using sparse-view filtered back projection for image reconstruction are not
fantastic. The reconstructions are blurry and contains streaking artifacts.  
There have been attempts to use neural networks to solve this problem before. Some use
the full filtered back projection as input and attempt to simply denoise the image. Others
use the raw data of a sparse-view sinogram and attempt to reconstruct the original image
from scratch. In this project, we will attempt to improve on such techniques by building a neural network that takes as input something that is more pre-processed than a sinogram,
but less so than a full filtered back projection.  

## Methods

### Data Generation
Our intention is to build a neural network that takes input data that is pre-processed differently than other attempts. Our input data is comprised of 16 single-view back projections. These 16 projections were generated using the MATLAB function “iradon” with only a single angle for each projection. We will refer to these 16 projections as the 16 channels of our image. We generated both filtered and unfiltered channels in this way. Each image is 64x64 pixels. The dataset we generated consists of a training set containing 10,000 sets of 16 unfiltered channels and 10,000 sets of 16 filtered channels. For testing we generated an additional 500 sets of each.  
We also generated sparse view sinograms (10,000 for training, 500 for testing) for comparison with previous work that use neural networks that take sinograms as input.  

### Neural Network
Our neural network is convolutional with 20 hidden layers using a 3x3 kernel. Each hidden
layer takes 64 channels of input and outputs 64 channels. The first layer is also convolutional
with a 5x5 kernel, taking 16 channels of input and outputting 64 channels. The last layer is
convolutional with a 3x3 kernel, taking 64 channels of input and outputting just one channel.  
For comparison with previous work, we also created a neural network that is identical
to our own, except that the first layer is a full linear layer in order to accept a sparse view
sinogram as input. This first layer takes in a 95x16 sinogram, and outputs 64 channels of
64x64 pixels each.  
For training, we used the Adam optimizer with an initial learning rate of .001. We cut
down this learning rate by a factor of ten every 37 epochs. We trained each network for two
hours on the BrownGPU computing cluster.

## Results
We trained our convolutional neural network architecture twice, once with the unfiltered
channels, and once with filtered channels; producing two trained networks with identical
architectures but different training results. We treat these as two separate neural networks,
despite the fact that their architectures are identical, because we will be comparing the
performance of using filtered and unfiltered channels as input.  
We trained each network for 153 epochs, the number of epochs was chosen such that the
network could be trained in 2 hours on BrownGPU. The unfiltered
channel network loss ended up just under 30%, while the filtered channel network loss ended
up just over 30%.  
We trained the network with the fully connected layer for only 8 epochs, the number of
epochs was chosen such that the network could be trained in 2 hours on BrownGPU. The sparse view sinogram network training loss ended up just under
50%, and the testing loss was worse.  

## Discussion
Both of our networks performed better than the fully connected layer network. Perhaps with
more epochs that network could have been trained better, however the important thing to
note here is how long it would have taken to train such a network to perform as well as ours.  
Comparing our two networks, we see that the network taking the unfiltered channels
actually performed slightly better. We think that this is due to the fact some information is
destroyed during the filtering pre-processing that the network then had to learn in addition
to what the other network was learning.  

## Appendix: Code
The code for our neural networks is contain in the following files (coded with PyTorch):  
- `MCB_code.py`
- `FMCB_code.py`
- `FCL_code.py`
The experimental code for the project is contained in `MCB_code.py` and `FMCB_code.py`. Our control group is the network whose code is contained in `FCL_code.py`.  
The code which generated out data with MATLAB is contained in:  
- `phantomProjectionDataset.m`,
which calls the functions contained in `FMCB.m` and `MCB.m`.

