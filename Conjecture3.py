# Imports
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import Lasso

# Load and preprocess the data including dropping rows with null values
car_data = pd.read_csv('./Car_sales.csv')
car_data = car_data.dropna()

# Create a new DataFrame with selected columns
data = pd.DataFrame().assign(
    wheelBase=car_data['Wheelbase'],
    curbWeight=car_data['Curb_weight'],
    engineSize=car_data['Engine_size'],
    length=car_data['Length'],
    width=car_data['Width'],
    fuelCapacity=car_data['Fuel_capacity']
)

# Select predictors (x) and response variable (y)
y = car_data.loc[:, ['Fuel_capacity']]
x = car_data.loc[:, ['Engine_size', 'Wheelbase', 'Width', 'Length', 'Curb_weight']]

# visualizing data using a pair plot
sns.pairplot(car_data, vars=['Price_in_thousands', 'Sales_in_thousands', '__year_resale_value', 'Engine_size',
                             'Horsepower', 'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity',
                             'Power_perf_factor'])

# visualizing data using a correlation plot
corMatrix = car_data[
    ['Price_in_thousands', 'Sales_in_thousands', '__year_resale_value', 'Engine_size', 'Horsepower', 'Wheelbase',
     'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Power_perf_factor']].corr()
heatmap = sns.heatmap(corMatrix, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap of our Response and Predictors', fontdict={'fontsize': 12}, pad=12)

print("--------Correlation Matrix-------")
print(corMatrix)
print("\n")


# checking P-test values using OLS
model = sm.OLS(y, x).fit()
print(model.summary())
print("\n")

# Split the data into features (x) and target variable (y)
x1 = data.loc[:, ['wheelBase', 'curbWeight', 'length', 'width', 'engineSize']]
y = data.loc[:, ['fuelCapacity']]

# Measure the accuracy of the model using Lasso regression
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x1, y, test_size=0.8)
reg = Lasso(alpha=1.0)
reg.fit(xTrain, yTrain)

# Measure accuracy using MSE
y_pred = reg.predict(xTest)
accuracy = reg.score(xTest, yTest)
print("Accuracy: " + str(accuracy))

# print pair-plot at the end
# plt.show()