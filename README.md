# Pneumonia Detection CNN Model

## **Overview**

This repository contains a **Convolutional Neural Network (CNN)** model for detecting pneumonia from X-ray images, implemented using TensorFlow in Google Colab. The project leverages a dataset obtained from Kaggle and demonstrates the application of data augmentation techniques and the Adam optimizer to achieve high accuracy.

## **Dataset**

The dataset used for this project is the [COVID-19 X-ray Dataset](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets) from Kaggle, which includes X-ray images categorized into two classes: **"NORMAL"** and **"PNEUMONIA"**.

## **Model Architecture**

The model is built using TensorFlow's Keras API and includes the following key components:
- **Convolutional Layers**: To extract features from the X-ray images.
- **MaxPooling Layers**: To downsample the feature maps and reduce dimensionality.
- **Data Augmentation**: Applied to increase the diversity of training data and improve model robustness.
- **Dropout**: To prevent overfitting by randomly dropping units during training.
- **Batch Normalization**: To stabilize and accelerate training by normalizing layer inputs.

## **Training**

The model is trained with the following parameters:
- **Batch Size**: 32
- **Image Size**: 128x128 pixels
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Epochs**: 10

Data augmentation techniques used include random flips, rotations, zooms, translations, and changes in height and width to enhance model generalization.

## **Results**

The trained model achieves a high accuracy on the validation set, demonstrating effective learning from a limited dataset through the use of data augmentation and optimized hyperparameters.

## **Usage**

To replicate the results or further experiment with the model, follow these steps:
1. **Clone the repository**:
    ```bash
    git clone https://github.com/ShehanPer/PNEUMONIA-DETECTION-CNN.git
    ```
2. **Install the required libraries**:
    ```bash
    pip install tensorflow opendatasets
    ```
3. **Run the notebook**:
    Open the provided Colab notebook and execute the cells to train the model and evaluate its performance.

## **Contributions**

This project was developed as part of self-learning from various tutorials and resources. Contributions to improve the model or provide feedback are welcome.

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
