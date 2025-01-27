#suraj -- giving all final outputs needed but worse prediction than before
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file into a DataFrame
df = pd.read_csv('soapnutshistory.csv')

# Generate hypothetical product prices for training purposes
df['Product Price'] = np.random.uniform(low=50, high=150, size=df.shape[0])

# Select the features (columns 2 to 7) and the target variable (product price)
X = df.iloc[:, 2:7]
y = df['Product Price'].astype(float)

# Handle missing values by imputing with the mean of the column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions on the testing set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Cross-validation
cv_scores = cross_val_score(best_model, X_imputed, y, cv=5, scoring='r2')
print("Cross-Validation R-squared scores:", cv_scores)
print("Mean Cross-Validation R-squared score:", np.mean(cv_scores))

# Predict future product price with sample future data
value1 = 0.0
value2 = 0.0
value3 = 0.0
value4 = 0.0
value5 = 100.0

future_data = pd.DataFrame([[value1, value2, value3, value4, value5]],
                           columns=["Organic Conversion Percentage",
                                    "Ad Conversion Percentage",
                                    "Total Profit",
                                    "Total Sales",
                                    "Predicted Sales"])

future_data_imputed = imputer.transform(future_data)

predicted_price = best_model.predict(future_data_imputed)
print("Predicted Future Product Price:", predicted_price)
