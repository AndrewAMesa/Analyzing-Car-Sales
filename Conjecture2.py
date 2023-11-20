
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model, metrics
import numpy as np



# Check 1: Price has a negative correlation with a car's sales

# Read in the data and drop empty rows
initialData = pd.read_csv('Car_sales.csv')
initialData = initialData.dropna()

# Select columns of interest
data = pd.DataFrame().assign(
    salesInThousands=initialData['Sales_in_thousands'],
    priceInThousands=initialData['Price_in_thousands']
)

# Split the data into features (x) and target variable (y)
x = data.loc[:, ['priceInThousands']]
y = data.loc[:, ['salesInThousands']]

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.8)

# Run the solver
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(xTrain.values, yTrain.values)

# Display slope and intercept
print("--------Check 1: Price vs. Sales--------")
print("Intercept:", reg.intercept_)
print("Coefficient (Beta_1):", reg.coef_)

# Plot the regression line
plotX = np.linspace(min(xTrain.values), max(xTrain.values), 100).reshape(-1, 1)
plotY = reg.predict(plotX)
plt.figure()
plt.plot(xTrain,yTrain,'ro')
plt.plot(xTest,yTest,'go')
plt.plot(plotX,plotY,'b-')
plt.xlabel("Price in thousands")
plt.title("Check 1: Regression Line")
#plt.show()

# Use the metrics package to print errors
print('Training Error:', metrics.mean_squared_error(yTrain, reg.predict(xTrain.values)))
print('Testing Error:', metrics.mean_squared_error(yTest, reg.predict(xTest.values)))
print("\n")



# Check 2: Horsepower has a positive correlation with a car's sales

# Select columns of interest
data = pd.DataFrame().assign(
    salesInThousands=initialData['Sales_in_thousands'],
    horsepower=initialData['Horsepower']
)

# Split the data into features (x) and target variable (y)
x = data.loc[:, ['horsepower']]
y = data.loc[:, ['salesInThousands']]

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.8)

# Run the solver
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(xTrain.values, yTrain.values)

# Display slope and intercept
print("--------Check 2: Horsepower vs. Sales--------")
print("Intercept:", reg.intercept_)
print("Coefficient (Beta_1):", reg.coef_)

# Plot the regression line
plotX = np.linspace(min(xTrain.values), max(xTrain.values), 100).reshape(-1, 1)
plotY = reg.predict(plotX)
plt.figure()
plt.plot(xTrain, yTrain, 'ro')
plt.plot(xTest, yTest, 'go')
plt.plot(plotX, plotY, 'b-')
plt.xlabel("Horsepower")
plt.ylabel("Sales in thousands")
plt.title("Check 2: Regression Line")
#plt.show()

# Use the metrics package to print errors
print('Training Error:', metrics.mean_squared_error(yTrain, reg.predict(xTrain.values)))
print('Testing Error:', metrics.mean_squared_error(yTest, reg.predict(xTest.values)))
print("\n")



# Check 3: Engine Size has a positive correlation with a car's sales

# Select columns of interest
data = pd.DataFrame().assign(
    salesInThousands=initialData['Sales_in_thousands'],
    engineSize=initialData['Engine_size']
)

# Split the data into features (x) and target variable (y)
x = data.loc[:, ['engineSize']]
y = data.loc[:, ['salesInThousands']]

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.8)

# Run the solver
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(xTrain.values, yTrain.values)

# Display slope and intercept
print("--------Check 3: Engine Size vs. Sales--------")
print("Intercept:", reg.intercept_)
print("Coefficient (Beta_1):", reg.coef_)

# Plot the regression line
plotX = np.linspace(min(xTrain.values), max(xTrain.values), 100).reshape(-1, 1)
plotY = reg.predict(plotX)
plt.figure()
plt.plot(xTrain, yTrain, 'ro')
plt.plot(xTest, yTest, 'go')
plt.plot(plotX, plotY, 'b-')
plt.xlabel("Engine Size")
plt.ylabel("Sales in thousands")
plt.title("Check 3: Regression Line")
#plt.show()

# Use the metrics package to print errors
print('Training Error:', metrics.mean_squared_error(yTrain, reg.predict(xTrain.values)))
print('Testing Error:', metrics.mean_squared_error(yTest, reg.predict(xTest.values)))