# Regression

# Notebook Content<br>

[Overview & Data](#Overview)

# Regression
[Linear Regression](#Linear-Regression)<br>
[Polynomial Regression](#Polynomial-Regression)<br>
[Quantile Regression](#Quantile-Regression)<br>
[Ridge Regression](#Ridge-Regression)<br>
[Lasso Regression](#Lasso-Regression)<br>
[Elastic Net Regression](#Elastic-Net-Regression)<br>
[Multiple Linear Regression](#Multiple-Linear-Regression)<br>
[Support Vector Regression](#Support-Vector-Regression)<br>
[Decision Tree - CART](#Decision-Tree---CART)<br>
[Random Forest Regression](#Random-Forest-Regression)<br>
[Gradient Boosting (GBM)](#Gradient-Boosting)<br>
[Stochastic Gradient Descent](#Stochastic-Gradient-Descent)<br>
[KNN Regressor](#KNN-Regressor)<br>
[XGBoost Regressor](#XGBoost-Regressor)<br>
[LightGBM](#LightGBM)<br>
[Regressors Report](#Regressors-Report)<br>


# Hyperparameter Tuning
[Grid-Search](#Grid-Search)<br>
[Random Search Cross Validation](#Random-Search-Cross-Validation)<br>
[Bayesian Optimization](#Bayesian-Optimization)<br>
[Bayesian optimization using hyperopt](#Bayesian-optimization-using-hyperopt)<br>
[Bayesian Optimization using Skopt](#Bayesian-Optimization-using-Skopt)<br>

# Overview

"In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a continues dependent variable and one or more independent variables. 
Following are the Regression Algorithms widely used -"				



```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
%matplotlib inline
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Linear Regression

In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a continues dependent variable and one or more independent variables.
In this Notebook, we will briefly study what linear regression is and how it can be implemented for both two variables and multiple variables using Scikit-Learn

The dataset we will be using here can be downloaded from - https://www.kaggle.com/zaraavagyan/weathercsv/data#

The dataset contains information on weather conditions recorded on each day at various weather stations around the world. Information includes precipitation, snowfall, temperatures, wind speed and whether the day included thunderstorms or other poor weather conditions.
So our task is to predict the maximum temperature taking input feature as the minimum temperature.

The following command imports the CSV dataset using pandas:


```python
dataset = pd.read_csv('dataset/weather.csv')
```

Let’s explore the data a little bit by checking the number of rows and columns in our datasets.


```python
dataset.shape
```




    (366, 22)



To see the statistical details of the dataset, we can use describe():


```python
dataset.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RISK_MM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>363.000000</td>
      <td>364.000000</td>
      <td>359.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
      <td>366.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.265574</td>
      <td>20.550273</td>
      <td>1.428415</td>
      <td>4.521858</td>
      <td>7.909366</td>
      <td>39.840659</td>
      <td>9.651811</td>
      <td>17.986339</td>
      <td>72.035519</td>
      <td>44.519126</td>
      <td>1019.709016</td>
      <td>1016.810383</td>
      <td>3.890710</td>
      <td>4.024590</td>
      <td>12.358470</td>
      <td>19.230874</td>
      <td>1.428415</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.025800</td>
      <td>6.690516</td>
      <td>4.225800</td>
      <td>2.669383</td>
      <td>3.481517</td>
      <td>13.059807</td>
      <td>7.951929</td>
      <td>8.856997</td>
      <td>13.137058</td>
      <td>16.850947</td>
      <td>6.686212</td>
      <td>6.469422</td>
      <td>2.956131</td>
      <td>2.666268</td>
      <td>5.630832</td>
      <td>6.640346</td>
      <td>4.225800</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.300000</td>
      <td>7.600000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>36.000000</td>
      <td>13.000000</td>
      <td>996.500000</td>
      <td>996.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>5.100000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.300000</td>
      <td>15.025000</td>
      <td>0.000000</td>
      <td>2.200000</td>
      <td>5.950000</td>
      <td>31.000000</td>
      <td>6.000000</td>
      <td>11.000000</td>
      <td>64.000000</td>
      <td>32.250000</td>
      <td>1015.350000</td>
      <td>1012.800000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.625000</td>
      <td>14.150000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.450000</td>
      <td>19.650000</td>
      <td>0.000000</td>
      <td>4.200000</td>
      <td>8.600000</td>
      <td>39.000000</td>
      <td>7.000000</td>
      <td>17.000000</td>
      <td>72.000000</td>
      <td>43.000000</td>
      <td>1020.150000</td>
      <td>1017.400000</td>
      <td>3.500000</td>
      <td>4.000000</td>
      <td>12.550000</td>
      <td>18.550000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.500000</td>
      <td>25.500000</td>
      <td>0.200000</td>
      <td>6.400000</td>
      <td>10.500000</td>
      <td>46.000000</td>
      <td>13.000000</td>
      <td>24.000000</td>
      <td>81.000000</td>
      <td>55.000000</td>
      <td>1024.475000</td>
      <td>1021.475000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>17.000000</td>
      <td>24.000000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20.900000</td>
      <td>35.800000</td>
      <td>39.800000</td>
      <td>13.800000</td>
      <td>13.600000</td>
      <td>98.000000</td>
      <td>41.000000</td>
      <td>52.000000</td>
      <td>99.000000</td>
      <td>96.000000</td>
      <td>1035.700000</td>
      <td>1033.200000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>24.700000</td>
      <td>34.500000</td>
      <td>39.800000</td>
    </tr>
  </tbody>
</table>
</div>



And finally, let’s plot our data points on a 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data using the below script :


```python
dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_16_0.png)


Our next step is to divide the data into “attributes” and “labels”.
Attributes are the independent variables while labels are dependent variables whose values are to be predicted. In our dataset, we only have two columns. We want to predict the MaxTemp depending upon the MinTemp recorded. Therefore our attribute set will consist of the “MinTemp” column which is stored in the X variable, and the label will be the “MaxTemp” column which is stored in y variable.


```python
X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)
```

Next, we split 80% of the data to the training set while 20% of the data to test set using below code.
The test_size variable is where we actually specify the proportion of the test set.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

After splitting the data into training and testing sets, finally, the time is to train our algorithm. For that, we need to import LinearRegression class, instantiate it, and call the fit() method along with our training data.


```python
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



The linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. To see the value of the intercept and slope calculated by the linear regression algorithm for our dataset, execute the following code.


```python
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
```

    [14.56202411]
    [[0.81953755]]
    

Now that we have trained our algorithm, it’s time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make predictions on the test data, execute the following script:


```python
y_pred = regressor.predict(X_test)
```

Now compare the actual output values for X_test with the predicted values, execute the following script:


```python
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25.2</td>
      <td>23.413030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.5</td>
      <td>13.086857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.1</td>
      <td>27.264856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22.2</td>
      <td>25.461874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.4</td>
      <td>26.937041</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>18.9</td>
      <td>20.216833</td>
    </tr>
    <tr>
      <th>70</th>
      <td>22.8</td>
      <td>27.674625</td>
    </tr>
    <tr>
      <th>71</th>
      <td>16.1</td>
      <td>21.446140</td>
    </tr>
    <tr>
      <th>72</th>
      <td>25.1</td>
      <td>24.970151</td>
    </tr>
    <tr>
      <th>73</th>
      <td>12.2</td>
      <td>14.070302</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 2 columns</p>
</div>



We can also visualize comparison result as a bar graph using the below script :
Note: As the number of records is huge, for representation purpose I’m taking just 25 records.


```python
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_30_0.png)


The final step is to evaluate the performance of the algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For regression algorithms, three evaluation metrics are commonly used:
Mean Absolute Error (MAE) is the mean of the absolute value of the errors. It is calculated as:

<img src="https://miro.medium.com/max/670/1*4kvomfLGxysM67hza_-B9Q.png">

2. Mean Squared Error (MSE) is the mean of the squared errors and is calculated as:

<img src="https://miro.medium.com/max/610/1*T37cOEU9OkXNPuqGQcXHSA.png">

3. Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

<img src="https://miro.medium.com/max/654/1*SGBsn7WytmYYbuTgDatIpw.gif">

Luckily, we don’t have to perform these calculations manually. The Scikit-Learn library comes with pre-built functions that can be used to find out these values for us.
Let’s find the values for these metrics using our test data.


```python
# change the input parameter 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

    Mean Absolute Error: 3.5094353112899594
    Mean Squared Error: 17.011877668640622
    Root Mean Squared Error: 4.124545753006096
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Polynomial regression

If your data points clearly will not fit a linear regression (a straight line through all data points), it might be ideal for polynomial regression.

Polynomial regression, like linear regression, uses the relationship between the variables x and y to find the best way to draw a line through the data points.

Download Eaxample dataset from - https://media.geeksforgeeks.org/wp-content/uploads/data.csv

Example
Start by drawing a scatter plot:


```python
# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Importing the dataset 
datas = pd.read_csv('dataset/data.csv')
datas
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sno</th>
      <th>Temperature</th>
      <th>Pressure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>40</td>
      <td>0.0060</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>60</td>
      <td>0.0300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>80</td>
      <td>0.0900</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>100</td>
      <td>0.2700</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Dividing dataset in X & y
X = datas.iloc[:, 1:2].values 
y = datas.iloc[:, 2].values 
```


```python
# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 

lin.fit(X, y) 

```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show() 
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_43_0.png)



```python
# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show() 
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_44_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Quantile Regression


Quantile regression is the extension of linear regression and we generally use it when outliers, high skeweness and heteroscedasticity exist in the data.


We first need to load some modules and to retrieve the data. Conveniently, the Engel dataset is shipped with statsmodels.


```python
%matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data = sm.datasets.engel.load_pandas().data
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>foodexp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>420.157651</td>
      <td>255.839425</td>
    </tr>
    <tr>
      <th>1</th>
      <td>541.411707</td>
      <td>310.958667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>901.157457</td>
      <td>485.680014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>639.080229</td>
      <td>402.997356</td>
    </tr>
    <tr>
      <th>4</th>
      <td>750.875606</td>
      <td>495.560775</td>
    </tr>
  </tbody>
</table>
</div>




```python
#The LAD model is a special case of quantile regression where q=0.5
mod = smf.quantreg('foodexp ~ income', data)
res = mod.fit(q=.5)
print(res.summary())
```

                             QuantReg Regression Results                          
    ==============================================================================
    Dep. Variable:                foodexp   Pseudo R-squared:               0.6206
    Model:                       QuantReg   Bandwidth:                       64.51
    Method:                 Least Squares   Sparsity:                        209.3
    Date:                Mon, 22 Jun 2020   No. Observations:                  235
    Time:                        07:59:47   Df Residuals:                      233
                                            Df Model:                            1
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     81.4823     14.634      5.568      0.000      52.649     110.315
    income         0.5602      0.013     42.516      0.000       0.534       0.586
    ==============================================================================
    
    The condition number is large, 2.38e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

We estimate the quantile regression model for many quantiles between .05 and .95, and compare best fit line from each of these models to Ordinary Least Squares results.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Ridge Regression


```python
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
```

In scikit-learn, a ridge regression model is constructed by using the Ridge class. The first line of code below instantiates the Ridge Regression model with an alpha value of 0.01. The second line fits the model to the training data.

Using the Weather dataset here


```python
#Ridge
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train) 

pred_test_rr= rr.predict(X_test)
```


```python
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_test_rr))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_test_rr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_test_rr)))
```

    Mean Absolute Error: 3.509436081243688
    Mean Squared Error: 17.011882182366772
    Root Mean Squared Error: 4.124546300184637
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Lasso Regression

In scikit-learn, a lasso regression model is constructed by using the Lasso class. The first line of code below instantiates the Lasso Regression model with an alpha value of 0.01. The second line fits the model to the training data.


```python
#Lasso
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train) 

pred_test_lasso= model_lasso.predict(X_test)
```


```python
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_test_lasso))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_test_lasso))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_test_lasso)))
```

    Mean Absolute Error: 3.5097096449126055
    Mean Squared Error: 17.013488931828462
    Root Mean Squared Error: 4.124741074519522
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Elastic Net Regression

In scikit-learn, an ElasticNet regression model is constructed by using the ElasticNet class. The first line of code below instantiates the ElasticNet Regression with an alpha value of 0.01. The second line fits the model to the training data.


```python
#Elastic Net
model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(X_train, y_train) 

pred_test_enet= model_enet.predict(X_test)
```


```python
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_test_enet))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_test_enet))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_test_enet)))
```

    Mean Absolute Error: 3.5096848564308916
    Mean Squared Error: 17.013343090308307
    Root Mean Squared Error: 4.124723395611918
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Multiple Linear Regression

<img src = "https://miro.medium.com/max/1238/1*r3aOsJoXHX7uC2nxn2lygQ.png">

Linear regression involving multiple variables is called “multiple linear regression” or multivariate linear regression. The steps to perform multiple linear regression are almost similar to that of simple linear regression. The difference lies in the evaluation. You can use it to find out which factor has the highest impact on the predicted output and how different variables relate to each other.

The dataset we will be using here can be downloaded from - https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

The dataset related to red variants of the Portuguese “Vinho Verde” wine. We will take into account various input features like fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. Based on these features we will predict the quality of the wine.

Lets get Started :


```python
dataset = pd.read_csv('dataset/winequality-red.csv')
```


```python
dataset.shape
```




    (1599, 12)




```python
dataset.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



Let us clean our data little bit, So first check which are the columns the contains NaN values in it :



```python
dataset.isnull().any()
```




    fixed acidity           False
    volatile acidity        False
    citric acid             False
    residual sugar          False
    chlorides               False
    free sulfur dioxide     False
    total sulfur dioxide    False
    density                 False
    pH                      False
    sulphates               False
    alcohol                 False
    quality                 False
    dtype: bool



Once the above code is executed, all the columns should give False, In case for any column you find True result, then remove all the null values from that column using below code.


```python
dataset = dataset.fillna(method='ffill')
```

Our next step is to divide the data into “attributes” and “labels”. X variable contains all the attributes/features and y variable contains labels.


```python
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']]
y = dataset['quality']
```

Let's check the average value of the “quality” column.


```python
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1eadec83668>




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_85_1.png)


Next, we split 80% of the data to the training set while 20% of the data to test set using below code.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Now lets train our model.


```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



As said earlier, in the case of multivariable linear regression, the regression model has to find the most optimal coefficients for all the attributes. To see what coefficients our regression model has chosen, execute the following script:


```python
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>0.041284</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>-1.149528</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>-0.177927</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>0.027870</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>-1.873407</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>0.002684</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>-0.002777</td>
    </tr>
    <tr>
      <th>density</th>
      <td>-31.516666</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.254486</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>0.924040</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>0.267797</td>
    </tr>
  </tbody>
</table>
</div>



Now let's do prediction on test data.


```python
y_pred_reg = regressor.predict(X_test)
```

Check the difference between the actual value and predicted value.


```python
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_reg})
df.head(25)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1109</th>
      <td>6</td>
      <td>5.782930</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>5</td>
      <td>5.036193</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>7</td>
      <td>6.596989</td>
    </tr>
    <tr>
      <th>487</th>
      <td>6</td>
      <td>5.339126</td>
    </tr>
    <tr>
      <th>979</th>
      <td>5</td>
      <td>5.939529</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>6</td>
      <td>5.007207</td>
    </tr>
    <tr>
      <th>542</th>
      <td>5</td>
      <td>5.396162</td>
    </tr>
    <tr>
      <th>853</th>
      <td>6</td>
      <td>6.052112</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>4</td>
      <td>4.867603</td>
    </tr>
    <tr>
      <th>412</th>
      <td>5</td>
      <td>4.950676</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>5</td>
      <td>5.285804</td>
    </tr>
    <tr>
      <th>475</th>
      <td>5</td>
      <td>5.412653</td>
    </tr>
    <tr>
      <th>799</th>
      <td>6</td>
      <td>5.705742</td>
    </tr>
    <tr>
      <th>553</th>
      <td>5</td>
      <td>5.129217</td>
    </tr>
    <tr>
      <th>1537</th>
      <td>6</td>
      <td>5.528852</td>
    </tr>
    <tr>
      <th>1586</th>
      <td>6</td>
      <td>6.380524</td>
    </tr>
    <tr>
      <th>805</th>
      <td>7</td>
      <td>6.810125</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>5</td>
      <td>5.738033</td>
    </tr>
    <tr>
      <th>1547</th>
      <td>5</td>
      <td>5.976188</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4</td>
      <td>5.086134</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>7</td>
      <td>6.344799</td>
    </tr>
    <tr>
      <th>549</th>
      <td>6</td>
      <td>5.164010</td>
    </tr>
    <tr>
      <th>1341</th>
      <td>6</td>
      <td>5.642040</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>4</td>
      <td>6.146290</td>
    </tr>
    <tr>
      <th>191</th>
      <td>6</td>
      <td>5.481780</td>
    </tr>
  </tbody>
</table>
</div>



Now let's plot the comparison of Actual and Predicted values


```python
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_97_0.png)


The final step is to evaluate the performance of the algorithm. We’ll do this by finding the values for MAE, MSE, and RMSE. Execute the following script:


```python
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_reg))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_reg))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))
```

    Mean Absolute Error: 0.46963309286611077
    Mean Squared Error: 0.38447119782012446
    Root Mean Squared Error: 0.6200574149384268
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Support Vector Regression

Support vector regression can solve both linear and non-linear models. SVM uses non-linear kernel functions (such as polynomial) to find the optimal solution for non-linear models.

SVMs assume that the data it works with is in a standard range, usually either 0 to 1, or -1 to 1 (roughly). So the normalization of feature vectors prior to feeding them to the SVM is very important. (This is often called whitening, although there are different types of whitening.) You want to make sure that for each dimension, the values are scaled to lie roughly within this range. Otherwise, if e.g. dimension 1 is from 0-1000 and dimension 2 is from 0-1.2, then dimension 1 becomes much more important than dimension 2, which will skew results.


```python
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X_train.values)
y = sc_y.fit_transform(y_train.values.reshape(-1, 1))
```


```python
#Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here
from sklearn.svm import SVR
# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
regressor = SVR(kernel='rbf')
regressor.fit(X,y)
```

    C:\Users\raghuram\Anaconda3\lib\site-packages\sklearn\utils\validation.py:760: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



We need to reverse transform our predictions as we have trained our model on scaled features


```python
# Now let's do prediction on test data.
y_pred_SVM = sc_y.inverse_transform ((regressor.predict (sc_X.transform(X_test.values))))
```


```python
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_SVM))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_SVM))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_SVM)))
```

    Mean Absolute Error: 0.43944155694818593
    Mean Squared Error: 0.37325969380095075
    Root Mean Squared Error: 0.6109498292011798
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Decision Tree - CART

Decision Tree Regression:
Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.

Let’s see the Step-by-Step implementation –


```python
# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  
  
# fit the regressor with X and Y data 
regressor.fit(X_train, y_train) 
```




    DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=0, splitter='best')




```python
# Now let's do prediction on test data.

y_pred_CART = regressor.predict(X_test) 
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_CART))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_CART))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_CART)))
```

    Mean Absolute Error: 0.459375
    Mean Squared Error: 0.621875
    Root Mean Squared Error: 0.7885905147793751
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Random Forest Regression

Random Forest is a collection of decision trees and average/majority vote of the forest is selected as the predicted output.


We import the random forest regression model from skicit-learn, instantiate the model, and fit (scikit-learn’s name for training) the model on the training data.


```python
# Import RF Regressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=1000, n_jobs=None, oob_score=False,
                          random_state=42, verbose=0, warm_start=False)




```python
# Now let's do prediction on test data.

y_pred_RF = rf.predict(X_test) 
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_RF))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_RF))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)))
```

    Mean Absolute Error: 0.40428437500000003
    Mean Squared Error: 0.318432278125
    Root Mean Squared Error: 0.5642980401569724
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Gradient Boosting

In gradient boosting, the ensemble model we try to build is also a weighted sum of weak learners

Boosting is a sequential technique which works on the principle of ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher. This technique is followed for a classification problem while a similar technique is used for regression.


```python
#Import GBM Algorithm
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(random_state=0)
# Train the model on training data
reg.fit(X_train, y_train)
```




    GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                              init=None, learning_rate=0.1, loss='ls', max_depth=3,
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=100,
                              n_iter_no_change=None, presort='deprecated',
                              random_state=0, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)




```python
# Now let's do prediction on test data.

y_pred_GBM = reg.predict(X_test) 
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_GBM))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_GBM))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_GBM)))
```

    Mean Absolute Error: 0.4633618705208503
    Mean Squared Error: 0.3699687048055903
    Root Mean Squared Error: 0.6082505279945019
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) regressor basically implements a plain SGD learning routine supporting various loss functions and penalties to fit linear regression models.



```python
# standardizing data

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
x_train = scaler.transform(X_train)
x_test=scaler.transform(X_test)

x_test=np.array(x_test)
y_test=np.array(y_test)
```


```python
#Importing SGD Regressor Library

import numpy as np
from sklearn.linear_model import SGDRegressor
n_iter=100
clf_ = SGDRegressor(max_iter=n_iter)
clf_.fit(x_train, y_train)
```




    SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
                 eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                 learning_rate='invscaling', loss='squared_loss', max_iter=100,
                 n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
                 shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
                 warm_start=False)




```python
# Now let's do prediction on test data.
y_pred_sksgd=clf_.predict(x_test)
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_sksgd))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_sksgd))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_sksgd)))
```

    Mean Absolute Error: 0.46946573119464663
    Mean Squared Error: 0.38557232127100044
    Root Mean Squared Error: 0.6209447006545755
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# KNN Regressor


K nearest neighbors is a simple algorithm that stores all available cases and predict the numerical target based on a similarity measure



```python
#Importing KNN Regressor
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                        weights='uniform')



We can make predict on the test data use knn regresson with n_neightbors = 3

We can analyse how accuracy gets affected by n_neighbors: We can use different value 3 n_neighbors, and explain where good value n_neighbors for model.


```python
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                        weights='uniform')




```python
# Now let's do prediction on test data.
y_pred_KNN =reg.predict(X_test)
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_KNN))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_KNN))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_KNN)))
```

    Mean Absolute Error: 0.565625
    Mean Squared Error: 0.5725694444444444
    Root Mean Squared Error: 0.7566831863101257
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# XGBoost Regressor


```python
 #We'll start by loading the required libraries.
import xgboost as xgb
```


```python
xgbr = xgb.XGBRegressor() 
print(xgbr)
```

    XGBRegressor(base_score=None, booster=None, colsample_bylevel=None,
                 colsample_bynode=None, colsample_bytree=None, gamma=None,
                 gpu_id=None, importance_type='gain', interaction_constraints=None,
                 learning_rate=None, max_delta_step=None, max_depth=None,
                 min_child_weight=None, missing=nan, monotone_constraints=None,
                 n_estimators=100, n_jobs=None, num_parallel_tree=None,
                 objective='reg:squarederror', random_state=None, reg_alpha=None,
                 reg_lambda=None, scale_pos_weight=None, subsample=None,
                 tree_method=None, validate_parameters=None, verbosity=None)
    


```python
#we'll fit the model with train data.
xgbr.fit(X_train, y_train)
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints='()',
                 n_estimators=100, n_jobs=0, num_parallel_tree=1,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                 validate_parameters=1, verbosity=None)




```python
# Now let's do prediction on test data.
y_pred_XGB =xgbr.predict(X_test)
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_XGB))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_XGB))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_XGB)))
```

    Mean Absolute Error: 0.41266453117132185
    Mean Squared Error: 0.4016609481817802
    Root Mean Squared Error: 0.6337672665748684
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# LightGBM


"Light GBM is a gradient boosting framework that uses tree based learning algorithm.
Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm."



```python
# lightgbm for regression
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(X_train, y_train)
```




    LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                  importance_type='split', learning_rate=0.1, max_depth=-1,
                  min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                  n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                  random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                  subsample=1.0, subsample_for_bin=200000, subsample_freq=0)




```python
# Now let's do prediction on test data.
y_pred_LGBM = model.predict(X_test)
```


```python
#Error Calculations

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_LGBM))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_LGBM))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_LGBM)))
```

    Mean Absolute Error: 0.43484010857597377
    Mean Squared Error: 0.37228440169007965
    Root Mean Squared Error: 0.6101511302047057
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Regressors Report


```python
d = [{"Algorithm" :'Linear Regression','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_reg),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_reg),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg))},
 {"Algorithm" :'Support Vector Regression','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_SVM),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_SVM),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_SVM))},
{"Algorithm" :'Decision Tree - CART','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_CART),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_CART),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_CART))},
 {"Algorithm" :'Random Forest Regression','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_RF),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_RF),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF))},
 {"Algorithm" :'Gradient Boosting (GBM)','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_GBM),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_GBM),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_GBM))},
 {"Algorithm" :'Stochastic Gradient Descent','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_sksgd),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_sksgd),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_sksgd))},
 {"Algorithm" :'KNN Regressor','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_KNN),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_KNN),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_KNN))},
 {"Algorithm" :'XGB Regressor','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_XGB),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_XGB),
 'Root Mean Squared Error':np.sqrt(metrics.mean_squared_error(y_test, y_pred_XGB))},
 {"Algorithm" :'LightGBM','Mean Absolute Error':metrics.mean_absolute_error(y_test, y_pred_LGBM),'Mean Squared Error':metrics.mean_squared_error(y_test, y_pred_LGBM),
 'Root Mean Squared Error':np.sqrt(np.sqrt(metrics.mean_squared_error(y_test, y_pred_LGBM)))}
]
```


```python
final = pd.DataFrame(d)
final_1 = final.sort_values('Mean Absolute Error')
final_1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>Mean Absolute Error</th>
      <th>Mean Squared Error</th>
      <th>Root Mean Squared Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Random Forest Regression</td>
      <td>0.404284</td>
      <td>0.318432</td>
      <td>0.564298</td>
    </tr>
    <tr>
      <th>7</th>
      <td>XGB Regressor</td>
      <td>0.412665</td>
      <td>0.401661</td>
      <td>0.633767</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM</td>
      <td>0.434840</td>
      <td>0.372284</td>
      <td>0.781122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Support Vector Regression</td>
      <td>0.439442</td>
      <td>0.373260</td>
      <td>0.610950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree - CART</td>
      <td>0.459375</td>
      <td>0.621875</td>
      <td>0.788591</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boosting (GBM)</td>
      <td>0.463362</td>
      <td>0.369969</td>
      <td>0.608251</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stochastic Gradient Descent</td>
      <td>0.469466</td>
      <td>0.385572</td>
      <td>0.620945</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.469633</td>
      <td>0.384471</td>
      <td>0.620057</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNN Regressor</td>
      <td>0.565625</td>
      <td>0.572569</td>
      <td>0.756683</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
#Visualization 
sns.set_color_codes("muted")
sns.barplot(x='Mean Absolute Error', y='Algorithm', data=final_1, color="r")

plt.xlabel('Mean Absolute Error')
plt.title('Regressor Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Mean Squared Error', y='Algorithm', data=final_1, color="g")

plt.xlabel('Mean Squared Error')
plt.title('Regressor Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Root Mean Squared Error', y='Algorithm', data=final_1, color="b")

plt.xlabel('Root Mean Squared Error')
plt.title('Regressor Accuracy')
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_162_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_162_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_162_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Hyperparameter Tuning

# Grid Search
Grid-searching is the process of scanning the data to configure optimal parameters for a given model. Depending on the type of model utilized, certain parameters are necessary. Grid-searching does NOT only apply to one model type. Grid-searching can be applied across machine learning to calculate the best parameters to use for any given model.

It is important to note that Grid-searching can be extremely computationally expensive and may take your machine quite a long time to run. Grid-Search will build a model on each parameter combination possible. It iterates through every parameter combination and stores a model for each combination.

**Note:** Here Grid search is demonstrated for only one model but it can be replicated across all the model with changing its respective hyperparameters


**Cross Validation**<br>
The technique of cross validation (CV) is best explained by example using the most common method, K-Fold CV. When we approach a machine learning problem, we make sure to split our data into a training and a testing set. In K-Fold CV, we further split our training set into K number of subsets, called folds. We then iteratively fit the model K times, each time training the data on K-1 of the folds and evaluating on the Kth fold (called the validation data). As an example, consider fitting a model with K = 5. The first iteration we train on the first four folds and evaluate on the fifth. The second time we train on the first, second, third, and fifth fold and evaluate on the fourth. We repeat this procedure 3 more times, each time evaluating on a different fold. At the very end of training, we average the performance on each of the folds to come up with final validation metrics for the model.


![CV.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/CV.PNG)

For hyperparameter tuning, we perform many iterations of the entire K-Fold CV process, each time using different model settings. We then compare all of the models, select the best one, train it on the full training set, and then evaluate on the testing set. This sounds like an awfully tedious process! Each time we want to assess a different set of hyperparameters, we have to split our training data into K fold and train and evaluate K times. If we have 10 sets of hyperparameters and are using 5-Fold CV, that represents 50 training loops.


```python
# Same data used for gird search as used in Random For multiple linear regression 
# dataset = pd.read_csv('winequality-red.csv')
X_train.shape, y_train.shape, X_test.shape
```




    ((1279, 11), (1279,), (320, 11))




```python
# PErforming Model without  Grid Search 

# Import RF Regressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)


# Now let's do prediction on test data.

y_pred_RF = rf.predict(X_test) 
```


```python
#Grid Search
from sklearn.model_selection import GridSearchCV
```


```python
# Create the parameter grid based 
# parameter value can be used according to data understanding 
param_grid = {
    'bootstrap': [True],
    'max_depth': [3,4,5],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100,500, 1000]
}
```


```python
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters
grid_search.best_params_
```

    Fitting 3 folds for each of 72 candidates, totalling 216 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:   14.5s
    [Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed:   56.9s
    [Parallel(n_jobs=-1)]: Done 216 out of 216 | elapsed:  1.4min finished
    




    {'bootstrap': True,
     'max_depth': 5,
     'max_features': 3,
     'min_samples_leaf': 4,
     'min_samples_split': 10,
     'n_estimators': 500}



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Random Search Cross Validation
Usually, we only have a vague idea of the best hyperparameters and thus the best approach to narrow our search is to evaluate a wide range of values for each hyperparameter


```python
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 6, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
```

    {'n_estimators': [10, 55, 100], 'max_features': ['auto', 'sqrt'], 'max_depth': [3, 4, 6, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}
    


```python
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations(n_iter), and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# Fit the random search model
rf_random.fit(X_train, y_train)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  58 tasks      | elapsed:    3.9s
    [Parallel(n_jobs=-1)]: Done 293 out of 300 | elapsed:   17.2s remaining:    0.3s
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:   17.8s finished
    




    RandomizedSearchCV(cv=3, error_score=nan,
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       ccp_alpha=0.0,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       max_samples=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100,
                                                       n_jobs=None, oob_score=False,
                                                       random_state=None, verbose=0,
                                                       warm_start=False),
                       iid='deprecated', n_iter=100, n_jobs=-1,
                       param_distributions={'bootstrap': [True, False],
                                            'max_depth': [3, 4, 6, None],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [10, 55, 100]},
                       pre_dispatch='2*n_jobs', random_state=42, refit=True,
                       return_train_score=False, scoring=None, verbose=2)




```python
rf_random.best_params_
```




    {'n_estimators': 55,
     'min_samples_split': 5,
     'min_samples_leaf': 2,
     'max_features': 'sqrt',
     'max_depth': None,
     'bootstrap': False}



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Bayesian Optimization
A popular alternative to tune the model hyperparameters is Bayesian Optimization. Bayesian Optimization is a probabilistic model-based technique used to find minimum of any function. This approach can yield better performance on the test set while it requires fewer iterations than random search. It takes into account past evaluations when choosing the optimal set of hyperparameters. Thus it chooses its parameter combinations in an informed way. In doing so, it focus on those parameters that yield the best possible scores. Thus, this technique requires less number of iterations to find the optimal set of parameter values. It ignores those areas of the parameter space that are useless. Hence, it is less time-consuming and not frustrating at all.

Bayesian optimization is also called **Sequential Model-Based Optimization (SMBO)**. It finds the value that minimizes an objective function by building a surrogate function. A **surrogate function** is nothing but a probability model based on past evaluation results of the objective. In the surrogate function, the input values to be evaluated are selected based on the criteria of expected improvement. Bayesian methods use past evaluation results to choose the next input values. So, this method excludes the poor input values and limit the evaluation of the objective function by choosing the next input values which have done well in the past.

There are a number of Python libraries that enable us to implement Bayesian Optimization for machine learning models. The examples of libraries are **Spearmint, Hyperopt or SMAC**. Scikit-learn also provides a library named Scikit-optimize for Bayesian optimization.

Bayesian Optimization methods differ in how they construct the surrogate function. Spearmint uses Gaussian Process surrogate while SMAC uses Random Forest Regression. Hyperopt uses the Tree Parzen Estimator (TPE) for optimization.

## Bayesian optimization using hyperopt

### Objective function
The aim is to minimize the objective function. It takes in a set of values as input (in this case hyperparameters of GBM model) and outputs a real value to minimize - the cross validation loss.
We will write the objective function for the GBM model with 5-fold cross validation.

In the objective-function, cross-validation is done. Once the cross validation is complete, we get the mean score. We want a value to minimize. So, we take negative of score. This value is then returned as the loss key in the return dictionary.
The objective function returns a dictionary of values - loss and status.
Next, we define the domain space.


### Domain space
The domain space is the range of values that we want to evaluate for each hyperparameter.In each iteration of the search, the Bayesian optimization algorithm will choose one value for each hyperparameter from the domain space. In Bayesian optimization this space has probability distributions for each hyperparameter value rather than discrete values. When first tuning a model, we should create a wide domain space centered around the default values and then refine it in subsequent searches.

### Optimization algorithm
Writing the optimization algorithm in hyperopt is very simple. It just involves a single line of code. We should use the Tree Parzen Estimator (tpe). The best parameter using Hyperopt for DT is shown below:


```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#!pip3 install scikit-optimize
#!pip3 install hyperopt
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before")

data = pd.read_csv('dataset/weather.csv')
data = data.dropna()
X = data[['MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine']]
y = data.MinTemp
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
def hyperopt_train_test(params):
    X_ = X[:]
    clf = DecisionTreeRegressor(**params)
#     cc = cross_val_score(clf, X, y).mean()
#     print(cc)
    return cross_val_score(clf, X, y).mean()
space4dt = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'criterion': hp.choice('criterion', ["mse"])
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)
print('best:', best)
# print best
```

    100%|█████████████████████████████████████████████| 300/300 [00:09<00:00, 31.25trial/s, best loss: 0.14782626484260714]
    best: {'criterion': 0, 'max_depth': 3, 'max_features': 2}
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Bayesian Optimization using Skopt

Bayesian optimization using Gaussian Processes.


```python
# import dataset
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

data = pd.read_csv('dataset/winequality-red.csv')
data = data.dropna()
X_ = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
X = X_.to_numpy()
y_ = data.quality
y = y_.to_numpy()

# gradient boosted trees tend to do well on problems like this
reg = GradientBoostingRegressor(n_estimators=50, random_state=0)
```


```python
from skopt.space import Real, Integer
from skopt.utils import use_named_args


# The list of hyper-parameters we want to optimize. For each one we define the
# bounds, the corresponding scikit-learn parameter name, as well as how to
# sample values from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(1, 5, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, 3, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set
# scikit-learn estimator parameters
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))
```


```python
import numpy as np
from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

"Best score=%.4f" % res_gp.fun
```




    'Best score=0.4961'




```python
print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                            res_gp.x[2], res_gp.x[3],
                            res_gp.x[4]))
```

    Best parameters:
    - max_depth=4
    - learning_rate=0.152645
    - max_features=3
    - min_samples_split=100
    - min_samples_leaf=96
    


```python
from skopt.plots import plot_convergence

plot_convergence(res_gp)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1eae5fe82b0>




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Regression/output_189_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

