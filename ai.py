import matplotlib as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
diabetes=datasets.load_diabetes()
print(diabetes.keys())
#for unifeatured graph
diabetes_x=diabetes.data[:,np.newaxis,2]
print(diabetes_x)
#slicing of data
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-30:]
diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]
model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted=model.predict(diabetes_x_test)
print ("mean squared error is",mean_squared_error(diabetes_y_test,diabetes_y_predicted))
wieght=print("weight_1",model.coef_)
print("intercept_1",model.intercept_)
#plotting a scatter plot
plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predicted)
plt.show()

#for multifeatured graph
diabetes_x=diabetes.data
print(diabetes_x)
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-30:]
diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]
model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted=model.predict(diabetes_x_test)
print ("mean squared error is",mean_squared_error(diabetes_y_test,diabetes_y_predicted))
new_weight = print("weight_2",model.coef_)
print("intercept_2",model.intercept_)

#difference of error between unifeatured and multifeatured wieghts
wieght_error = wieght-new_weight
print("reduction in error is" + wieght_error)