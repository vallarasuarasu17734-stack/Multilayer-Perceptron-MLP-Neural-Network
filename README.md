# Experiment 5 : Multilayer Perceptron (MLP) Neural Network

## Aim

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> **Note:** The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP
> experiment.

## Overview

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural
network consisting of an input layer, one or more hidden layers, and an
output layer. It learns patterns in data by adjusting weights and biases
during training using backpropagation.

In this experiment, the MLP model is trained on the **Digits dataset**
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## Algorithm

1.  **Start.**
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. **Stop.**
ry slightly with changes to the train-test split,
    random seed, or model parameters.


## Program

    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load Digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create MLP classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    )
    
    # Train the model
    mlp.fit(X_train, y_train)
    
    # Predict test data
    y_pred = mlp.predict(X_test)
    
    # Calculate accuracy
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training loss curve
    plt.plot(mlp.loss_curve_)
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()





## output
    Test Accuracy: 0.9805555555555555
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        33
               1       1.00      1.00      1.00        28
               2       1.00      1.00      1.00        33
               3       0.97      0.97      0.97        34
               4       0.98      1.00      0.99        46
               5       0.98      0.96      0.97        47
               6       0.97      0.97      0.97        35
               7       1.00      0.97      0.99        34
               8       0.97      1.00      0.98        30
               9       0.95      0.95      0.95        40
    
        accuracy                           0.98       360
       macro avg       0.98      0.98      0.98       360
    weighted avg       0.98      0.98      0.98       360

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/8749bfcd-c6b0-4254-837d-f7aa88f42331" />




## Result
Thus, the Multilayer Perceptron (MLP) Neural Network was successfully implemented using Python and Scikit-learn. The model was trained on the handwritten Digits dataset and evaluated using test accuracy and a classification report. The training loss curve was plotted to visualize the learning process. The model successfully classified handwritten digits from 0 to 9.
