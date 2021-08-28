# Neural_Network_Charity_Analysis

![Deep_Learning](https://github.com/ysbcode/Neural_Network_Charity_Analysis/blob/main/Resources/Deep_Learning.png?raw=true)

## Overview 

This project explores Alphabet Soup's charity data to classify the success of charitable donations using deep-learning neural networks with the TensorFlow library in Python. Data was analyzed using the following steps: 

1. Preprocessing data for the neural networks model
2. Compiling, training, and evaluating the model
3. Optimizing the model 

Using my knowledge from machine learning and neural networks for this project I use the features in this dataset I create a binary classifier that will tell the customer whether or not the applicants will be successful using alphabet soup. The dataset contains over 34,000 organizations that have been funded by alphabet soup. There are a number of columns that capture the metadata of each organization such as organization type, the use of funding amongst others.

## Results 

Here is what I found after using deep learning the analyze the dataset: 

### Data Preprocessing

What variable(s) are considered the target(s) for your model?
- The `IS_SUCCESSFUL` column was our target variable for the deep learning model, as it contains binary data referring to whether or not the charity donation was successful
What variable(s) are considered to be the features for your model?
- I used the following columns as features for the model: `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT`
What variable(s) are neither targets nor features, and should be removed from the input data?
- The first step was to remove `EIN` and `NAME` columns from the input data, as they were not relevant to our analysis

![Columns Used](https://github.com/ysbcode/Neural_Network_Charity_Analysis/blob/main/Resources/Columns%20Used.PNG?raw=true)

### Compiling, Training, and Evaluating the Model 

![CTE1](https://github.com/ysbcode/Neural_Network_Charity_Analysis/blob/main/Resources/CTE1.PNG?raw=true)
![CTE1](https://github.com/ysbcode/Neural_Network_Charity_Analysis/blob/main/Resources/CTE2.PNG?raw=true)

- As we can see from the image, the neural network model has two hidden layers with 80 neurons in the first layer and 30 in the second \
I used the `ReLU` activation function for the hidden layers, and `Sigmoid` for the output layer 
- The original model accuracy was about 73%, which is not optimal to predict the outcome of the charity donations 
- To improve the performance of the model, I did the following: \
Applied binning to the `ASK_AMT` feature to organize the values by set intervals \
Added more hidden layers to the model \
I also used a different activation function in the hidden layers (`tanh`), which gave us the highest accuracy score out of all other attempts, but still under 75%

## Summary

Even after all attempts to optimize the model, the accuracy did not reach our minimum requirement of 75%. Therefore, we cannot successfully use this model to classify charitable donations. Due to the fact that the target variable is binary, making this a binary classification, supervised machine learning could be useful in trying to classify the success of charitable donations, rather than deep learning. Perhaps the Random Forest Classifier could be used in this model to create decision trees that could classify the target variable.
