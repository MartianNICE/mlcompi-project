#suraj -- final code till now
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file into a DataFrame
df = pd.read_csv('soapnutshistory.csv')

# Generate hypothetical product prices for training purposes
# Replace NaN values in the 'Product Price' column with hypothetical values
df['Product Price'] = np.random.uniform(low=50, high=150, size=df.shape[0])

# Select the features (columns 2 to 7) and the target variable (product price)
X = df.iloc[:, 2:7]  # Features (columns 3 to 7)
y = df['Product Price'].astype(float)  # Convert target variable to float

# Check for NaN values in the features
print("Number of NaN values in each column:\n", X.isnull().sum())

# Handle missing values by imputing with the mean of the column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Verify that there are no NaN values after imputation
print("Shape of X_imputed:", X_imputed.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Replace these example values with the actual future values you have
value1 = 0.0  # Example: organic conversion percentage
value2 = 0.0  # Example: ad conversion percentage
value3 = 0.0  # Example: total profits
value4 = 0.0  # Example: total sales
value5 = 100.0  # Example: predicted sales

# Construct the future data array with feature names
future_data = pd.DataFrame([[value1, value2, value3, value4, value5]],
                           columns=["Organic Conversion Percentage",
                                    "Ad Conversion Percentage",
                                    "Total Profit",
                                    "Total Sales",
                                    "Predicted Sales"])

# Impute missing values in future data
future_data_imputed = imputer.transform(future_data)

# Predict future product price
predicted_price = model.predict(future_data_imputed)
print("Predicted Future Product Price:", predicted_price)
