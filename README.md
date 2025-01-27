ML Compi Project
Overview
This project contains three different code snippets aimed at predicting product prices using machine learning models on historical sales data of soapnuts. The snippets feature models such as Random Forest Regressor, Linear Regression, and an implementation of Q-Learning for dynamic pricing.

Project Structure
Random Forest Regressor with GridSearchCV

Linear Regression Model

Q-Learning for Dynamic Pricing

Random Forest Regressor with GridSearchCV
This code snippet focuses on using a Random Forest Regressor model to predict product prices. It includes hyperparameter tuning using GridSearchCV and evaluates the model using mean squared error and r-squared metrics.

Steps:

Load the sales data from a CSV file.

Generate hypothetical product prices for training.

Select features and target variables.

Handle missing values with mean imputation.

Split the data into training and testing sets.

Initialize and tune the Random Forest Regressor model.

Make predictions and evaluate the model.

Predict future product prices with sample data.

Linear Regression Model
This snippet employs a Linear Regression model to predict product prices, handling missing values through mean imputation. It evaluates the model using mean squared error and r-squared metrics and predicts future prices based on provided feature values.

Steps:

Load the historical sales data.

Generate hypothetical product prices.

Select features and target variables.

Handle missing values.

Split data into training and testing sets.

Train the Linear Regression model.

Make predictions and evaluate the model.

Predict future product prices using sample future data.

Q-Learning for Dynamic Pricing
This implementation uses Q-Learning to optimize dynamic pricing strategies based on a simulated sales environment. It involves creating a pricing environment and training a Q-Learning agent to maximize rewards through price adjustments.

Steps:

Load and preprocess historical sales data.

Handle missing values and split the data.

Train a Random Forest Regressor model to simulate sales based on price.

Define a Q-Learning environment with simulated sales.

Implement a Q-Learning agent for dynamic pricing.

Train the agent through episodes.

Test the trained agent.

How to Run
Ensure you have the required libraries installed:

sh
pip install numpy pandas scikit-learn
Place your soapnutshistory.csv file in the same directory as your script.

Run the desired code snippet in a Python environment.

Feel free to modify the code and experiment with different model parameters or additional features to improve the predictions.


