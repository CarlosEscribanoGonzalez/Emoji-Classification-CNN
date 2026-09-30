## Overview
Convolutional Neural Network built in PyTorch to classify emoji images into five categories: airplanes, birds, boats, rabbits and felines. The dataset is small and the final test set is private, so the focus of the project is on reducing overfitting and training a network that generalizes rather than memorizes.

## Problem
* Input: small emoji images with three color channels
* Output: one of five classes (airplanes, birds, boats, rabbits, felines)
* Very limited dataset, with some classes having far fewer samples than others
* Success is measured by how well the model generalizes, tracked through the validation loss

## Approach

**Data pipeline**
* Random split into training, validation and (in early stages) testing subsets
* Images and labels converted to tensors and wrapped in `DataLoader`s
* Custom `AugmentedDataset` (subclass of `torch.utils.data.Dataset`) that applies a transform on the fly, simulating a much larger training set

**Data augmentation**
* Horizontal and vertical flips
* Random rotations and translations
* Brightness and contrast changes
* Transformations kept mild so augmented images stay close to the original appearance

**Model**
* Padding in the convolutions, to avoid losing information too quickly on small images
* Four convolutional layers before the first fully connected layer
* Batch normalization before each convolution, for more stable training and regularization
* Dropout to prevent the network from relying on specific activations

**Training**
* Loss: `CrossEntropyLoss`
* Optimizer: Adam, which works better than SGD on small datasets
* Scheduler: `CosineAnnealingLR`, to lower the learning rate as the model converges
* 75 epochs, the point at which the validation loss stabilizes
* The model with the lowest validation loss is kept, so any late overfitting does not affect the final result
* No early stopping, since Adam and data augmentation make the validation loss oscillate
* Training and validation loss curves are plotted at the end of each run

## Results
* Validation loss close to **0.1** in the best runs (0.1071 in the best recorded one)
* **~95% accuracy** on an external test set
* Results vary between runs (validation loss around 0.3 in the worst cases, with some overfitting), because both the train/validation split and the data augmentation are random
<p align = "center">
  <img width="620" height="472" alt="training and validation loss" src="https://github.com/user-attachments/assets/5182b2c6-2bed-4158-b4c3-0ed02b8de539" />
</p>

## Usage
* Open the notebook in Jupyter or Google Colab
* Run all cells; the training and validation loss curves are shown at the end
