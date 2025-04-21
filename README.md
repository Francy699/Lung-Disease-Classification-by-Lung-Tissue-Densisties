# Lung-Disease-Classification-by-Lung-Tissue-Densisties
The Lung Disease Classification project uses 100,000+ chest X-ray images from Kaggle to train a DenseNet121 model for predicting diseases like pneumonia, tuberculosis, and COVID-19. Built with TensorFlow/Keras and Streamlit, it offers real-time predictions via image uploads, optimized with transfer learning and deployed on Heroku.
# Lung Diseases Classification by Analysis of Lung Tissue Densities

This project focuses on the classification of lung diseases by analyzing lung tissue densities using advanced computational techniques. The goal is to assist in the early detection and diagnosis of lung diseases through automated analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Lung diseases are a significant health concern worldwide. This project leverages machine learning and image processing techniques to classify lung diseases based on the analysis of lung tissue densities from medical images.

## Features

- Preprocessing of lung tissue density data.
- Implementation of machine learning models for classification.
- Visualization of results and performance metrics.
- Support for multiple lung disease categories.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn
- **Tools**: Jupyter Notebook, Visual Studio Code

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lung-disease-classification.git
   cd lung-disease-classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset and place it in the appropriate directory.
2. Run the preprocessing script to clean and prepare the data.
3. Train the model using the training script:
   ```bash
   python train_model.py
   ```
4. Evaluate the model using the evaluation script:
   ```bash
   python evaluate_model.py
   ```

## Dataset

The dataset used for this project should contain labeled medical images of lung tissues. Ensure the dataset is preprocessed and split into training, validation, and testing sets.

## Model Training

The project uses a convolutional neural network (CNN) Densenet121 for image classification. The training script includes hyperparameter tuning and model checkpointing for optimal performance.

## Results

The results of the model, including accuracy, precision, recall, and F1-score, are visualized using Matplotlib. Confusion matrices and ROC curves are also generated.

