import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# The Data Set: 
hours = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
marks = np.array([35,40,50,55,65,70,80,85,90,95])

#model training: 
model = LinearRegression()
model.fit(hours, marks)

# Ask user for input
user_hours = float(input("Enter number of study hours: "))

# Machine Learning Prediction: 
predicted = model.predict([[user_hours]])
print("Predicted Marks:", predicted[0])

# for visualization:
plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()