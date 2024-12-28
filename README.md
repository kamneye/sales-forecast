# sales-forecast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Load the sales data
data = pd.read_csv('app_sales_data.csv')

# Check the first few rows of the data
print(data.head())
# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Set the 'date' column as the index of the DataFrame
data.set_index('date', inplace=True)

# Check for missing values
print(data.isnull().sum())

# Fill missing values with forward fill method if any
data.fillna(method='ffill', inplace=True)
# Extract additional time-based features
data['day'] = data.index.day
data['month'] = data.index.month
data['year'] = data.index.year
data['day_of_week'] = data.index.dayofweek

# Preview the data with additional features
print(data.head())
# Features and target variable
X = data[['day_of_week', 'month', 'year']]
y = data['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Sales', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Sales', color='red', linestyle='--')
plt.title('App Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Example of future data to predict (next 7 days)
future_dates = pd.date_range('2024-01-01', periods=7, freq='D')
future_data = pd.DataFrame({
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month,
    'year': future_dates.year
}, index=future_dates)

# Predict future sales
future_sales = model.predict(future_data)

# Show future predictions
future_sales_df = pd.DataFrame(future_sales, index=future_dates, columns=['predicted_sales'])
print(future_sales_df)
