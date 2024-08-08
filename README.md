## Project Summary: Cat vs Dog Classifier with ResNet50 Fine-Tuning
### App Link: https://resnet-image-classifier.streamlit.app/
### Objective
The objective of this project is to develop a robust image classifier that can distinguish between images of cats and dogs. By leveraging the ResNet50 pre-trained convolutional neural network (CNN) model and fine-tuning it with additional convolutional and dense layers, we aim to achieve high classification accuracy.

## Dataset
The dataset used for this project is the popular Cats vs Dogs dataset, which consists of 20,000 labeled images of cats and dogs, divided equally into 10,000 images per class.

## Methodology
### Data Preprocessing:

### Image Resizing:
Resize all images to 224x224 pixels to match the input size expected by ResNet50.

### Normalization:
Scale pixel values to the range [0,1] by dividing by 255.

### Data Augmentation:
Apply random transformations such as rotations, flips, and zooms to augment the training dataset and reduce overfitting.
Model Architecture:

## Base Model:
Utilize the ResNet50 architecture pre-trained on the ImageNet dataset, excluding its top fully connected layers.
### Custom Layers for Fine-Tuning:
3. Add a flatten layer to convert the 2D matrix to a 1D vector.
4. Add 1 fully connected dense layer with 1024 units and ReLU activation.
5. Include 1 dropout layers with a dropout rate of 0.2 after each dense layer to reduce overfitting.
6. Add a final dense layer with 1 unit and sigmoid activation for binary classification.
## Training:

### Loss Function:
Use binary cross-entropy loss.
### Optimizer:
Use the Stochastic Gradient Descent optimizer with default learning rate.
### Training Parameters:
Train the model for 10 epochs with a batch size of 20, using a validation split of 20% to monitor performance on the validation set.
Evaluation:

## Results
The fine-tuned ResNet50 model achieved a training accuracy of approximately 99.6% and a test accuracy of approximately 98.8%, indicating its effectiveness in distinguishing between cats and dogs.
