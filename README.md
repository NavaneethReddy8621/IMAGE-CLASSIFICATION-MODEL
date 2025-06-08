# IMAGE-CLASSIFICATION-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: NAVANEETH REDDY ADDLA

INTERN ID : CT04DF1817

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH

DESCRIPTION :
My project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch.It includes proper training, validation, and testing pipelines along with performance visualization and evaluation metrics.The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The goal was to train a CNN that can accurately classify these images.

My project includes:
- A custom CNN architecture with 3 convolutional layers.
- Data preprocessing and normalization.
- Train-validation-test split.
- Training with live tracking of loss and accuracy.
- Evaluation using classification report and test accuracy.
- Visualization of training progress.
- Model saving for future inference.

Technologies Used

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn (for classification report)

Dataset:
The CIFAR-10 dataset is automatically downloaded using `torchvision.datasets.CIFAR10`. No manual setup is required.

Commands to run project:

->pip install torch torchvision matplotlib scikit-learn
->python main.py

The model is saved as 'enhanced_cifar10_cnn.pth' after training.
Training and validation accuracy/loss are plotted at the end of training.
Final model is evaluated on the test set using:
Accuracy,Precision,Recall,F1-score
You can expect a test accuracy of 70-75% after 15 epochs depending on the machine and training conditions.

Here is a sample output for better understanding:

Epoch 1: Train Acc: 43.20%, Val Acc: 51.25%
Epoch 2: Train Acc: 58.31%, Val Acc: 63.87%
Test Accuracy: 72.84%



