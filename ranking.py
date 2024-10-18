#kernal_restart
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data for speed and density
speed = np.array([90, 80, 65, 54, 35])  # Speed in km/h
density = np.array([7, 27, 38, 50, 65])  # Density in veh/km

# Reshape the data for linear regression
X = density.reshape(-1, 1)  # Independent variable (density)
y = speed  # Dependent variable (speed)

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Free flow speed (v_f) is the intercept of the line
v_f = intercept

# Jam density (k_j) is where speed is zero, so solve for k_j: 0 = v_f - (v_f / k_j) * k_j
k_j = -intercept / slope

# Capacity is given by (v_f * k_j) / 4
capacity = (v_f * k_j) / 4

# Output results
v_f, k_j, capacity


# model_fit

import numpy

def test1():
    # Load the dataset
    X = numpy.array([1, 2, 3, 4, 10])
    y = numpy.array([2, 4, 6, 8, 10])
    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Train the model
    model = numpy.polyfit(X_train, y_train, 1)

    # Make predictions on the testing set
    y_pred = numpy.polyval(model, X_test)

    # Calculate the accuracy of the model

    accuracy = numpy.mean(y_pred == y_test)
    print("Accuracy:", accuracy)
test1()