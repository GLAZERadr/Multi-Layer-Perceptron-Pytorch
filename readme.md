# Multi Layer Perceptron Using Pytorch

## Introduction

In this project, we will explore the implementation of a Multi Layer Perceptron (MLP) using PyTorch. MLP is a type of feedforward neural network that consists of multiple layers of nodes (neurons) connected in a sequential manner. It is a versatile and widely used architecture that can be applied to various machine learning tasks, including classification, regression, and pattern recognition.

The goal of this project is to build and train an MLP model to perform a specific task, such as image classification or time series prediction. We will leverage the power of PyTorch, a popular deep learning framework, to construct and train our MLP model efficiently.

Throughout the project, we will cover important aspects such as data preprocessing, model architecture design, training process, and evaluation of the model's performance. We will also discuss key considerations and techniques for optimizing the MLP model and improving its accuracy.

By the end of this project, you will have a solid understanding of how to implement an MLP using PyTorch and be equipped with the knowledge to apply it to your own machine learning tasks. So let's dive in and explore the world of Multi Layer Perceptrons!

## Library

**Dependencies** used in this model is:

* matplotlib
* numpy
* pandas
* scikit-learn
* seaborn
* torch

## Dataset

Dataset used in this model is form factbook data. Factbook data contains a wealth of information about various countries across the globe. It covers a wide range of topics including economic indicators, demographic statistics, geographic details, social factors, and more. This comprehensive dataset provides valuable insights into different aspects of countries, allowing for in-depth analysis and comparisons.

For our project, we will specifically focus on a subset of Factbook data related to economic factors such as import and export data, GDP (Gross Domestic Product), investment, unemployment rate, and industrial production growth rate. These economic indicators play a crucial role in understanding a country's economic performance, trends, and potential areas of growth or concern.

* **Independant Data**: exports, imports, investment, unemployement rate, industrial production growth rate.

* **Dependant Data**: GDP.

## Model

The model used in this code is a Multi-Layer Perceptron (MLP) implemented using PyTorch. The MLP consists of three fully connected layers. The input layer has 5 neurons, representing the 5 input features. The hidden layers have 64 and 32 neurons respectively, and ReLU activation functions are applied after each hidden layer. The output layer has 1 neuron, which provides the predicted output.

The forward method defines the forward pass of the MLP, where the input data is propagated through the layers to generate the output. The layers are defined using the nn.Sequential module, which allows for a sequential arrangement of the layers.

During training, the L1 loss function (nn.L1Loss) is used to compute the loss between the predicted outputs and the target values. The Adagrad optimizer (torch.optim.Adagrad) is utilized with a learning rate of 1e-4 to update the model parameters and minimize the loss. The training process is performed for 5 epochs, with each epoch consisting of iterations over the training data.

The loss after each mini-batch is printed to monitor the training progress. At the end of each epoch, the current loss is reset. Finally, the completion of the training process is indicated.

Overall, this model architecture, loss function, and optimizer settings aim to train the MLP to predict the target variable based on the input features and minimize the L1 loss during the training process.

## Conclusion

**In conclusion**, our implementation of the Multi Layer Perceptron (MLP) using PyTorch for predicting GDP based on economic indicators from the Factbook dataset yielded mixed results. The model architecture consisted of three fully connected layers with ReLU activation, and training was performed using the L1 loss function and Adagrad optimizer.

While the model showed some predictive capability, as indicated by a relatively low Mean Squared Error (MSE), the R2 Score was quite low, suggesting limited explanatory power. Further refinement of the model, exploration of alternative architectures or optimization algorithms, and inclusion of additional features could potentially improve performance.

In summary, while our MLP model provides a foundation for predicting GDP based on economic indicators, there is room for improvement and further experimentation to enhance its accuracy and predictive power.

## Reference

[1] PyTorch: https://pytorch.org/
[2] Factbook: https://www.cia.gov/the-world-factbook/