# Classification of Brain MRI using Transfer Learning and PyTorch 🧠

## About The Project
In this project, we aim to classify MRI images of the brain using transfer learning and PyTorch. We use a pre-trained ResNet-50 model and change the last layer in different ways to find the best combination that fits our problem.

- #### Dataset
We use a publicly available dataset of MRI images of the brain on [kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), which consists of 3164 images of brain. We split the dataset into training (87%) and testing (23%) sets.

- #### Transfer Learning
We use a pre-trained ResNet-50 model as our base model and change the last layer in different ways to fine-tune the model for our specific problem. We experiment with different combinations of the last layer and choose the best one based on the validation accuracy.

- #### Hyperparameters
We experiment with different hyperparameters such as learning rate, optimizer, and number of epochs to find the best combination that maximizes the accuracy. We use the Adam optimizer with a learning rate of 0.001 and train the model for 20 epochs.

- #### Training and Testing
We train and test the model with the hyperparameters we choose and evaluate its performance on the testing set. We use cross-entropy loss as our loss function and monitor the training and validation accuracy to ensure that the model is not overfitting.

- #### Saving the Model
Finally, we save the final model for future use. We also provide instructions on how to load the saved model and use it to classify new MRI images.

### Library we used in this project:
- Numpy
- Matplotlib
- PyTorch
