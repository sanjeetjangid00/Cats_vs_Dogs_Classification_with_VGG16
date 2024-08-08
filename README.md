## Project Summary: Cat vs Dog Classifier with Vgg16 Fine-Tuning
### App Link: https://cat-vs-dog-vgg16.streamlit.app/
### Objective
The objective of this project is to develop a robust image classifier that can distinguish between images of cats and dogs. By leveraging the Vgg16 pre-trained convolutional neural network (CNN) model and fine-tuning it with additional convolutional and dense layers, we aim to achieve high classification accuracy.

## Dataset
The dataset used for this project is the popular Cats vs Dogs dataset, which consists of 20,000 labeled images of cats and dogs, divided equally into 10,000 images per class.

## Methodology
### Data Preprocessing:

### Image Resizing:
Resize all images to 150x150 pixels to match the input size expected by Vgg16.

### Normalization:
Scale pixel values to the range [0,1] by dividing by 255.

### Data Augmentation:
Apply random transformations such as rotations, flips, and zooms to augment the training dataset and reduce overfitting.
Model Architecture:

## Base Model:
Utilize the Vgg16 architecture pre-trained on the ImageNet dataset, excluding its top fully connected layers.
### Custom Layers for Fine-Tuning:
1. Add a flatten layer to convert the 2D matrix to a 1D vector.
2. Add 2 fully connected dense layers with 512 and 1024 units and ReLU activation.
3. Include 2 dropout layers with a dropout rate of 0.2 after each dense layer to reduce overfitting.
4. Add a final dense layer with 1 unit and sigmoid activation for binary classification.
## Training:

### Loss Function:
Use binary cross-entropy loss.
### Optimizer:
Use the Stochastic Gradient Descent optimizer with default learning rate.
### Training Parameters:
Train the model for 12 epochs with a batch size of 32, using a validation split of 20% to monitor performance on the validation set.
Evaluation:

## Results
The fine-tuned Vgg16 model achieved a training accuracy of approximately 97% and a test accuracy of approximately 96%, indicating its effectiveness in distinguishing between cats and dogs.
