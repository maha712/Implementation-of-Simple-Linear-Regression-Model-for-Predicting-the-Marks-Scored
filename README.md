# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:K.Mahalakshmi
RegisterNumber: 212222240057 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="blue")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,reg.predict(X_test),color="black")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output
Dataset
![dataset](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/e1e40ef4-6010-499c-b8a7-a5bd3f4be07a)
Head values
![head values](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/b79038ce-3175-418e-95be-7a105f5f6434)
Tail values
![Tail values](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/e77a39c7-6049-4a8b-a20a-df7564c41131)
X and Y values
![Tail values](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/50516ba7-afe4-4055-b080-0b3cc3b3aca6)

Predication values of X and Y
![prediction of x and y values](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/160cbb96-395d-42e0-92b6-c868372ceebf)
MSE,MAE and RMSE
![MSE,MAE and RMSE](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/331ff6cd-529e-495e-9056-4565e96f98ab)
Training Set
![ml ex 2](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/e664353c-de9f-489a-923c-3b7199293ffb)

![ml exp 2](https://github.com/maha712/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121156360/89b40d7c-950b-45e2-9370-fde13db4ac0d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
