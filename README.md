# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, numpy, matplotlib, and scikit-learn.
2. Load the dataset student_scores.csv into a DataFrame and print it to verify contents.
3. Display the first and last few rows of the DataFrame to inspect the data structure.
4. Extract the independent variable ( x ) and dependent variable ( y ) as arrays from the
DataFrame.
5. Split the data into training and testing sets, with one-third used for testing and a
fixed random_state for reproducibility.
6. Create and train a linear regression model using the training data.
7. Make predictions on the test data and print both the predicted and actual values for
comparison.
8. Plot the training data as a scatter plot and overlay the fitted regression line to visualize the
model's fit.
9. Plot the test data as a scatter plot with the regression line to show model performance on
unseen data.
10. Calculate and print error metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and
Root Mean Squared Error (RMSE) for evaluating model accuracy.
11. Display the plots to visually assess the regression results.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Shruthi D.N
RegisterNumber: 212223240155
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color='black')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("HOURS VS SCORES (TRAINING SET)")
plt.xlabel("HOURS")
plt.ylabel("SCORES")
plt.show()
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("HOURS VS SCORES (TRAINING SET)")
plt.xlabel("HOURS")
plt.ylabel("SCORES")
plt.show()
MAE=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",MAE)
MSE= mean_squared_error(Y_test,Y_pred)
print("MSE = ",MSE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
```

## Output:
![image](https://github.com/user-attachments/assets/47baeb57-6308-4b84-a9b5-84dd3714e1e7)

![image](https://github.com/user-attachments/assets/15fc0e1e-c955-4b0e-98a0-5d58db0f5ee1)

![image](https://github.com/user-attachments/assets/5a83282b-7d39-49f6-90f6-7a67daae4681)

![image](https://github.com/user-attachments/assets/3477b0ac-21c1-491f-a823-db26a08b3dbe)

![image](https://github.com/user-attachments/assets/53f2fa81-1b70-4dee-a2a9-8f0a3c4c1f19)

![image](https://github.com/user-attachments/assets/e3a9f0cb-9b56-4931-ac69-7cdb67cd2485)

![image](https://github.com/user-attachments/assets/5007406a-ae86-4538-9d37-0ea810bf1113)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
