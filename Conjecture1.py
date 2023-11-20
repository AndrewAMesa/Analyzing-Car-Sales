# Import necessary libraries
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
data = pd.read_csv('Car_sales.csv')

# Create a new DataFrame with selected columns and normalize the data
data = pd.DataFrame().assign(
    priceInThousands=data['Price_in_thousands'],
    horsepower=data['Horsepower'],
    fuelCapacity=data['Fuel_capacity'],
    fuelEfficiency=data['Fuel_efficiency'],
    powerPerfFactor=data['Power_perf_factor']
)

data = (data - data.min()) / (data.max() - data.min())  # Min-max normalization
data = data.dropna()  # Drop any rows with missing values

# Calculate the correlation matrix
corMatrix = data.corr()
round(corMatrix, 2)
print("--------Correlation Matrix-------")
print(corMatrix)
print("\n")

# Create a pair plot for visualization
pairplot = sns.pairplot(data, vars=["priceInThousands", "horsepower", "fuelCapacity", "fuelEfficiency", "powerPerfFactor"])

# Split the data into features (x) and target variable (y)
x = data.loc[:, ['horsepower', 'fuelCapacity', 'fuelEfficiency', 'powerPerfFactor']]
y = data.loc[:, ['priceInThousands']]

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.8)

# Create a linear regression model and fit it to the training data
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(xTrain, yTrain)

# Display the intercept and coefficients of the model
print("--------Intercepts and Coefficients-------")
print("Intercept: " + str(reg.intercept_))
print("Coefficients: " + str(reg.coef_) + "\n")

# Evaluate the model on training and testing data
print("--------Training and Testing Errors-------")
print('Training Error')
print(metrics.mean_squared_error(yTrain, reg.predict(xTrain)))
print('Testing Error')
print(metrics.mean_squared_error(yTest, reg.predict(xTest)))
print("\n")

# Calculate and display the R-squared value
rSquared = reg.score(xTrain, yTrain)
print("-------R Squared Value-------")
print((rSquared))

# print pair-plot at the end
# plt.show()