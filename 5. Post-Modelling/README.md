# Model Evaluation

# Notebook Content

# Evaluation Metrics for Regression

[Explained Variance Score](#Explained-Variance-Score)<br>
[Max error](#Max-error)<br>
[Mean Absolute Error](#Mean-Absolute-Error)<br>
[Mean Squared Error](#Mean-Squared-Error)<br>
[Mean Squared Logarithmic Error](#Mean-Squared-Logarithmic-Error)<br>
[Median Absolute Error](#Median-Absolute-Error)<br>
[R² score](#1)<br>
[Mean Percentage Error](#Mean-Percentage-Error)<br>
[Mean Absolute Percentage Error](#Mean-Absolute-Percentage-Error)<br>
[Weighted Mean Absolute Percentage Error](#Weighted-Mean-Absolute-error)<br>
[Tips for Metric Selection](#Tips-for-Metric-Selection)<br>
[Cross Validation](#Cross-Validation)<br>
[StratifiedKFold](#StratifiedKFold)<br>
    
# Evaluation Metrics for Binary Classification
    
[Cohen’s kappa](#2)<br>
[Hamming Loss](#Hamming-Loss)<br>
[Confusion Matrix](#Confusion-Matrix)<br>
[Precision-Recall Curve](#5)<br>
[ROC (Receiver Operating Characteristics)](#3)<br>
[AUC (Area Under the Curve)](#4)<br>
    
# Evaluation Metrics for Multi-Class Classification
 
[Accuarcy Score](#Accuarcy-Score)<br>
[Confusion Matrix for MultiClass](#Confusion-Matrix-for-MultiClass)<br>
[ROC and AUC](#ROC-and-AUC)<br>
[Precision Recall Curve for Multiclass](#Precision-Recall-Curve-for-Multiclass)<br>


# Evaluation Metrics for Regression
The **sklearn.metrics module implements several loss, score, and utility functions to measure regression performance**. Some of those have been enhanced to handle the **multioutput case**: mean_squared_error, mean_absolute_error, explained_variance_score and r2_score.These functions have an **multioutput keyword argument which specifies the way the scores or losses for each individual target should be averaged**. The default is 'uniform_average', which specifies a uniformly weighted mean over outputs.<br> 
If an ndarray of shape (n_outputs,) is passed, then its entries are interpreted as weights and an according weighted average is returned. If multioutput is 'raw_values' is specified, then all unaltered individual scores or losses will be returned in an array of shape (n_outputs,).
<br><br>The **r2_score and explained_variance_score accept an additional value 'variance_weighted'** for the multioutput parameter. This option leads to a weighting of each individual score by the variance of the corresponding target variable. This setting quantifies the globally captured unscaled variance. If the target variables are of different scale, then this score puts more importance on well explaining the higher variance variables. multioutput='variance_weighted' is the default value for r2_score for backward compatibility. This will be changed to uniform_average in the future.<br><br>

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Explained Variance Score
The explained_variance_score **computes the explained variance regression score**. **Explained variation measures the proportion to which a mathematical model accounts for the variation (dispersion)** of a given data set. Often, variation is quantified as variance; then, the more specific term explained variance can be used.<br><br>The best possible score is 1.0, lower values are worse.

If <img src="https://render.githubusercontent.com/render/math?math=hat{y}_i"> is the estimated target output, *y* the corresponding (correct) target output, and *Var* is Variance, the square of the standard deviation, then the explained variance is estimated as follow:<br>

<img src="https://render.githubusercontent.com/render/math?math=explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}"><br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Max error
The max_error function computes the **maximum residual error** , a metric that captures the worst case error between the predicted value and the true value. In a perfectly fitted single output regression model, max_error would be 0 on the training set and though this would be highly unlikely in the real world, this metric shows the extent of error that the model had when it was fitted.


If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the max error is defined as:<br>
<img src="https://render.githubusercontent.com/render/math?math={Max Error}(y, \hat{y}) = max(| y_i - \hat{y}_i |)"><br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>


## Mean Absolute Error
The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the absolute error loss or $l1$-norm loss.In statistics, mean absolute error (MAE) is a **measure of errors between paired observations** expressing the same phenomenon.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the i-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the mean absolute error (MAE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as:<br>

<img src="https://render.githubusercontent.com/render/math?math=\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|">.<br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Mean Squared Error
The mean_squared_error function computes mean square error, a risk metric **corresponding to the expected value of the squared (quadratic) error or loss**.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the mean squared error (MSE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as

<img src="https://render.githubusercontent.com/render/math?math=\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2"><br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>


## Mean Squared Logarithmic Error
The mean_squared_log_error function computes a **risk metric corresponding to the expected value of the squared logarithmic (quadratic) error or loss.**

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the mean squared logarithmic error (MSLE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as:<br>

<img src="https://render.githubusercontent.com/render/math?math=\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2"><br>
Where <img src="https://render.githubusercontent.com/render/math?math=\log_e (x)"> means the natural logarithm of *x*. <br>
**This metric is best to use when targets having exponential growth, such as population counts**, average sales of a commodity over a span of years etc. Note that this metric penalizes an under-predicted estimate greater than an over-predicted estimate.
<br><br><a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Median Absolute Error
The median_absolute_error is particularly interesting because **it is robust to outliers**. The loss is calculated by taking the **median of all absolute differences between the target and the prediction**.<br><br>
If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample and <img src="https://render.githubusercontent.com/render/math?math=y_i"> is the corresponding true value, then the median absolute error (MedAE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as<br>

<img src="https://render.githubusercontent.com/render/math?math=\text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid)">

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

<a id = '1'></a>

## R² score
The r2_score function **computes the coefficient of determination**, usually denoted as R².<br><br>
It represents the **proportion of variance (of y) that has been explained by the independent variables** in the model. It provides an **indication of goodness of fit** and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.<br><br>
As such variance is dataset dependent, **R² may not be meaningfully comparable across different datasets**. <br>Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of 0.0.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value for total $n$ samples, the estimated R² is defined as:

<img src="https://render.githubusercontent.com/render/math?math=R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}"> <br>
where <img src="https://render.githubusercontent.com/render/math?math=\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ and $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \epsilon_i^2"><br>


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Mean Percentage Error
In statistics, the mean percentage error (MPE) is the computed average of percentage errors by which forecasts of a model differ from actual values of the quantity being forecast.

The formula for the mean percentage error is:
![MPE.svg](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/Mean%20Percentage%20Error.svg) <br>

where <img src="https://render.githubusercontent.com/render/math?math=a_t"> is the actual value of the quantity being forecast, <img src="https://render.githubusercontent.com/render/math?math=f_t"> is the forecast, and n is the number of different times for which the variable is forecast.

Because actual rather than absolute values of the forecast errors are used in the formula, positive and negative forecast errors can offset each other; as a result the formula can be used as a measure of the bias in the forecasts.

A disadvantage of this measure is that it is undefined whenever a single actual value is zero.<br>
The fuction defination of MPE is given below


```python
def MPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) / y_true)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Mean Absolute Percentage Error
It is a simple average of absolute percentage errors.
where <img src="https://render.githubusercontent.com/render/math?math=A_t"> is the actual value and <img src="https://render.githubusercontent.com/render/math?math=F_t"> is the forecast value. The MAPE is also sometimes reported as a percentage, which is the above equation multiplied by 100. The difference between <img src="https://render.githubusercontent.com/render/math?math=A_t"> and <img src="https://render.githubusercontent.com/render/math?math=F_t"> is divided by the actual value At again. <br>
It **cannot be used if there are zero values** (which sometimes happens for example in demand data) because there would be a division by zero.<br>
For forecasts which are too low the percentage error cannot exceed 100%, but **for forecasts which are too high there is no upper limit to the percentage error**.<br>

The fuction defination of MAPE is Given below:


```python
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Weighted Mean Absolute error
The wMAPE is the metric in which the sales are weighted by sales volumne. Weighted Mean Absolute Percentage Error, as the name suggests, is a **measure that gives greater importance to faster selling products**. Thus it overcomes one of the potential drawbacks of MAPE. There is a very simple way to calculate WMAPE. This involves adding together the absolute errors at the detailed level, then calculating the total of the errors as a percentage of total sales.  This method of calculation leads to the additional benefit that it is robust to individual instances when the base is zero, thus overcoming the divide by zero problem that often occurs with MAPE.

WMAPE is a highly useful measure and is becoming increasingly popular both in corporate KPIs and for operational use. It is easily calculated and gives a concise forecast accuracy measurement that can be used to summarise performance at any detailed level across any grouping of products and/or time periods. If a measure of accuracy required this is calculated as 100 - WMAPE.

Now, in order to show how to run the different Accuracy metrics, we will be prforming Random Forest Regression on a dataset the download link for which is given below:

https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009<br>
We will start by importing the necessary libraries and the required dataset.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>



```python
# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```


```python
data = pd.read_csv('dataset/winequality-red.csv')
```


```python
data = data.fillna(method='ffill')
```

Our next step is to divide the data into “attributes” and “labels”. X variable contains all the attributes/features and y variable contains labels.


```python
X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']]
y = data['quality']
```

Next, we split 80% of the data to the training set while 20% of the data to test set using below code.


```python
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Now we will run random forest regression on the training set.


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

y_pred = rf.predict(X_test) 
```

### _Code_
Now we can calculate the accuracy of our model using the different accuracy metrics that we have explained above. 


```python
#Error Calculations
from sklearn import metrics
#Explained Variance Score
print("_____________________________________")
print('Explained variance Score:', metrics.explained_variance_score(y_test, y_pred, multioutput='raw_values'))
explained_variance_score= metrics.explained_variance_score(y_test, y_pred, multioutput='raw_values')
#Max Error
print("_____________________________________")
print('Max Error:', metrics.max_error(y_test, y_pred))
max_error=metrics.max_error(y_test, y_pred)
#Mean Absolute Error
print("_____________________________________")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
#Mean Squared Error
print("_____________________________________")
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
mean_squared_error=metrics.mean_squared_error(y_test, y_pred)
#Root Mean Squared Error
print("_____________________________________")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
root_mean_squared_error=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#Mean Squared Log Error
print("_____________________________________")
print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred))
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
#Median Absolute Error
print("_____________________________________")
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred))
median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
#Median Absolute Error
print("_____________________________________")
print('Mean Percentage Error:', MPE(y_test, y_pred))
mean_percentage_error=MPE(y_test, y_pred)
#Median Absolute Error
print("_____________________________________")
print('Mean Absolute Percentage Error:', MAPE(y_test, y_pred))
mean_absolute_percentage_error=MAPE(y_test, y_pred)
```

    _____________________________________
    Explained variance Score: [0.45337001]
    _____________________________________
    Max Error: 2.615
    _____________________________________
    Mean Absolute Error: 0.40428437500000003
    _____________________________________
    Mean Squared Error: 0.318432278125
    _____________________________________
    Root Mean Squared Error: 0.5642980401569724
    _____________________________________
    Mean Squared Log Error: 0.008019925336798525
    _____________________________________
    Median Absolute Error: 0.3089999999999997
    _____________________________________
    Mean Percentage Error: -0.024683098958333333
    _____________________________________
    Mean Absolute Percentage Error: 7.658587425595238
    


```python
#Dataframe of errors given by all the metrics explained above
data={"Error Metric":['Explained variance Score', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','Mean Squared Log Error','Median Absolute Error','Mean Percentage Error','Mean Absolute Percentage Error'],
      "Score":["%.4f" % explained_variance_score, "%.4f" % max_error, "%.4f" % mean_absolute_error, "%.4f" % mean_squared_error, "%.4f" % root_mean_squared_error, "%.4f" % mean_squared_log_error, "%.4f" % median_absolute_error, "%.4f" % mean_percentage_error, "%.4f" % mean_absolute_percentage_error]}
error_df = pd.DataFrame(data)
```


```python
error_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Error Metric</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Explained variance Score</td>
      <td>0.4534</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Max Error</td>
      <td>2.6150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mean Absolute Error</td>
      <td>0.4043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mean Squared Error</td>
      <td>0.3184</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Root Mean Squared Error</td>
      <td>0.5643</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mean Squared Log Error</td>
      <td>0.0080</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Median Absolute Error</td>
      <td>0.3090</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mean Percentage Error</td>
      <td>-0.0247</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mean Absolute Percentage Error</td>
      <td>7.6586</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Tips for Metric Selection
- The **MAE is also the most intuitive of the metrics** one should by just looking at the absolute difference between the data and the model’s predictions<br>


- MAE does not indicate underperformance or overperformance of the model (whether or not the model under or overshoots actual data). Each residual contributes proportionally to the total amount of error, meaning that larger errors will contribute linearly to the overall error<br> 


- A **small MAE** suggests the model is great at prediction, while a **large MAE** suggests that, model may have trouble in certain areas. A MAE of 0 means that your model is a perfect predictor of the outputs (but this will almost never happen)<br><br>


- While the MAE is easily interpretable, using the absolute value of the residual often is not as desirable as squaring this difference. Depending on how the model to treat outliers, or extreme values, in the data, one may want to bring more attention to these outliers or downplay them. The issue of outliers can play a major role in which error metric you use<br>


- Because of squaring the difference, the **MSE will almost always be bigger than the MAE**. For this reason, we cannot directly compare the MAE to the MSE. **one can only compare model’s error metrics to those of a competing model**<br>

- The effect of the square term in the MSE equation is most apparent with the presence of outliers in our data. **While each residual in MAE contributes proportionally to the total error, the error grows quadratically in MSE**<br><br>


- RMSE is the square root of the MSE. Because the **MSE is squared, its units do not match that of the original output**. Researchers will often use **RMSE to convert the error metric** back into similar units, making interpretation easier<br>


- Taking the square root before they are averaged, **RMSE gives a relatively high weight to large errors**, so RMSE should be useful when large errors are undesirable.<br><br>


- Just as MAE is the average magnitude of error produced by your model, the **MAPE is how far the model’s predictions are off** from their corresponding outputs on average. Like MAE, MAPE also has a clear interpretation since percentages are easier for people to conceptualize. **Both MAPE and MAE are robust to the effects of outliers**<br>


- Many of MAPE’s weaknesses actually stem from use division operation. MAPE is **undefined for data points where the value is 0**. Similarly, the MAPE can grow unexpectedly large if the actual values are exceptionally small themselves<br>


- Finally, the MAPE is **biased towards predictions** that are systematically less than the actual values themselves. That is to say, MAPE will be lower when the prediction is lower than the actual compared to a prediction that is higher by the same amount. 

The table given below can also be helpful in regard to metric understanding:


|Acroynm|Full Name|Residual Operation?|Robust To Outliers|
|-------|---------|-------------------|------------------|
|MAE|Mean Absolute Error|Absolute Value|Yes|
|MSE|Mean Squared Error	Square|No|No|
|RMSE|Root Mean Squared Error|Square|No|
|MAPE|Mean Absolute Percentage Error|Absolute Value|Yes|
|MPE|Mean Percentage Error|N/A|Yes|


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Cross Validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

**K-Folds Cross Validation**
<br>In K-Folds Cross Validation we split our data into k different subsets (or folds). We use k-1 subsets to train our data and leave the last subset (or the last fold) as test data. We then average the model against each of the folds and then finalize our model. After that we test it against the test set.


```python
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
```


```python
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
#Train the model on training data
rf.fit(X_train, y_train)
y_pred_kfold = rf.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_kfold)) 
mean_squared_error=metrics.mean_squared_error(y_test, y_pred_kfold)

```

    Mean Squared Error: 0.39817868338557993
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# StratifiedKFold
StratifiedKFold is a variation of KFold. First, StratifiedKFold shuffles your data, after that splits the data into n_splits parts and Done. Now, it will use each part as a test set. Note that it only and always shuffles data one time before splitting.


**Difference in KFold and ShuffleSplit output**

KFold will divide your data set into prespecified number of folds, and every sample must be in one and only one fold. A fold is a subset of your dataset.

ShuffleSplit will randomly sample your entire dataset during each iteration to generate a training set and a test set. The test_size and train_size parameters control how large the test and training test set should be for each iteration. Since you are sampling from the entire dataset during each iteration, values selected during one iteration, could be selected again during another iteration.

Summary: ShuffleSplit works iteratively, KFold just divides the dataset into k folds.

**Difference when doing validation**

In KFold, during each round you will use one fold as the test set and all the remaining folds as your training set. However, in ShuffleSplit, during each round n you should only use the training and test set from iteration n. As your data set grows, cross validation time increases, making shufflesplits a more attractive alternate. If you can train your algorithm, with a certain percentage of your data as opposed to using all k-1 folds, ShuffleSplit is an attractive option.


```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,y):
    # Split train-test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
```


```python
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=100, n_jobs=None, oob_score=False,
                          random_state=42, verbose=0, warm_start=False)




```python
y_pred_kfold = rf.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_kfold)) 
mean_squared_error=metrics.mean_squared_error(y_test, y_pred_kfold)

```

    Mean Squared Error: 0.4155485893416928
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Accuracy Metrics for Binary Classification

The True and Fake news classification dataset will be used to show how model evaluation for Binary Classification Can be done. The dataset can be downloaded from the link below:<br>
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset


```python
true = pd.read_csv(r'dataset/True.csv')
fake = pd.read_csv(r'dataset/fake.csv')
```


```python
fake['target'] = 'fake'
true['target'] = 'true'
news = pd.concat([fake, true]).reset_index(drop = True)
print('The size of dataset',news.shape)
```

    The size of dataset (44898, 5)
    


```python
news.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump Sends Out Embarrassing New Year’...</td>
      <td>Donald Trump just couldn t wish all Americans ...</td>
      <td>News</td>
      <td>December 31, 2017</td>
      <td>fake</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drunk Bragging Trump Staffer Started Russian ...</td>
      <td>House Intelligence Committee Chairman Devin Nu...</td>
      <td>News</td>
      <td>December 31, 2017</td>
      <td>fake</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sheriff David Clarke Becomes An Internet Joke...</td>
      <td>On Friday, it was revealed that former Milwauk...</td>
      <td>News</td>
      <td>December 30, 2017</td>
      <td>fake</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>
      <td>On Christmas day, Donald Trump announced that ...</td>
      <td>News</td>
      <td>December 29, 2017</td>
      <td>fake</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pope Francis Just Called Out Donald Trump Dur...</td>
      <td>Pope Francis used his annual Christmas Day mes...</td>
      <td>News</td>
      <td>December 25, 2017</td>
      <td>fake</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=2020)
# y_test_copy =y_test.copy()
```


```python
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
```

    accuracy: 99.05%
    


```python
from sklearn.metrics import accuracy_score
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
```

    accuracy: 99.05%
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id = '2'></a>

# Cohen’s kappa 

It is a statistic that **measures inter-annotator agreement**.

This function computes Cohen’s kappa, a **score that expresses the level of agreement between two annotators** on a classification problem. It is defined as:<br>
<img src="https://render.githubusercontent.com/render/math?math=\kappa = (p_o - p_e) / (1 - p_e)"><br>
where  is the empirical probability of agreement on the label assigned to any sample (the observed agreement ratio), and  is the expected agreement when both annotators assign labels randomly.  is estimated using a per-annotator empirical prior over the class labels


```python
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, prediction)

```




    0.9810118237654452



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Hamming Loss
The hamming_loss computes the average Hamming loss or **Hamming distance between two sets of samples**.

If <img src="https://render.githubusercontent.com/render/math?math=hat{y}_j"> is the predicted value for the *j*-th label of a given sample, <img src="https://render.githubusercontent.com/render/math?math=y_j"> is the corresponding true value, and <img src="https://render.githubusercontent.com/render/math?math=n_{labels}"> is the number of classes or labels, then the Hamming loss <img src="https://render.githubusercontent.com/render/math?math=L_{Hamming}"> between two samples is defined as:<br>
<img src="https://render.githubusercontent.com/render/math?math=L_{Hamming}(y, \hat{y}) = \frac{1}{n_\text{labels}} \sum_{j=0}^{n_\text{labels} - 1} 1(\hat{y}_j \not= y_j)"><br>
where $1(x)$ is the indicator function.




```python
from sklearn.metrics import hamming_loss
hamming_loss(y_test, prediction)

```




    0.009465478841870824



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>



# Confusion Matrix

A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix. The confusion matrix shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

![confusion%20matrix.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/confusion%20matrix.PNG)

Here,

<br>Class 1 : Positive
<br>Class 2 : Negative

**Definition of the Terms:**

<br>Positive (P) : Observation is positive (for example: is an apple).
<br>Negative (N) : Observation is not positive (for example: is not an apple).
<br>True Positive (TP) : Observation is positive, and is predicted to be positive.
<br>False Negative (FN) : Observation is positive, but is predicted negative.
<br>True Negative (TN) : Observation is negative, and is predicted to be negative.
<br>False Positive (FP) : Observation is negative, but is predicted positive.


```python
#using plot_confusion_matrix
from sklearn.metrics import confusion_matrix
table2=confusion_matrix(y_test, prediction)
#using seaborn
# fake = 0, true = 1
import seaborn as sns
ax = sns.heatmap(table2,annot=True,cmap='Blues', fmt='g')
ax.set(xlabel="Predticted label", ylabel = "True label")
ax.set_title('Confusion Matix')
```




    Text(0.5, 1, 'Confusion Matix')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_66_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id = '5'></a>

# Precision Recall (PR) Curve
A PR curve is simply a graph with **Precision values on the y-axis and Recall values on the x-axis**. In other words, the PR curve contains TP/(TP+FN) on the y-axis and TP/(TP+FP) on the x-axis.

1. It is important to note that Precision is also called the Positive Predictive Value (PPV).
2. Recall is also called Sensitivity, Hit Rate or True Positive Rate (TPR).


```python
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(model, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve')
```




    Text(0.5, 1.0, '2-class Precision-Recall curve')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_69_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id = '3'></a>

# ROC (Receiver Operating Characteristics)

ROC Curves are used to see how well your classifier can separate positive and negative examples and to identify the best threshold for separating them.

In ROC curves, the true positive rate (TPR, y-axis) is plotted against the false positive rate (FPR, x-axis). These quantities are defined as follows:<br>
**TPR(True Positve Rate) = TP/(TP+FN)<br>
FPR(False Positve Rate) = FP/(TN+FP)**



```python
pre = pd.Series(prediction)
y_test.replace(to_replace = 'true',value=1,inplace = True)
y_test.replace(to_replace = 'fake',value= 0,inplace = True)
pre.replace(to_replace = 'fake',value= 0,inplace = True,)
pre.replace(to_replace = 'true',value= 1,inplace = True)
```


```python
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
```

    0.9905486227211209
    


```python
import matplotlib.pyplot as plt
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_76_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id = '4'></a>

# AUC (Area Under the Curve)
The model performance is determined by looking at the area under the ROC curve (or AUC). **An excellent model has AUC near to the 1.0**, which means it has a good measure of separability. For your model, the AUC is the combined are of the blue, green and purple rectangles,
<br>so the AUC = 0.4 x 0.6 + 0.2 x 0.8 + 0.4 x 1.0 = 0.80.

![AUC.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/AUC.PNG)


```python
# AUC explaned in above illustration 
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
```

    0.9905486227211209
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>



# Accuracy Metrics for Multi-Class Classification
In multiclass and multilabel classification task, the notions of precision, recall, and F-measures can be applied to each label independently. There are a few ways to combine results across labels, specified by the average argument to the average_precision_score (multilabel only), f1_score, fbeta_score, precision_recall_fscore_support, precision_score and recall_score functions, as described above. <br><br>
Note that if all labels are included, “micro”-averaging in a multiclass setting will produce precision, recall and $F$ that are all identical to accuracy. Also note that “weighted” averaging may produce an F-score that is not between precision and recall.

## Accuarcy Score
The accuracy_score function **computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions**.

**In multilabel classification, the function returns the subset accuracy**. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample and <img src="https://render.githubusercontent.com/render/math?math=y_i"> is the corresponding true value, then the fraction of correct predictions over <img src="https://render.githubusercontent.com/render/math?math=n_\text{samples}"> is defined as:<br>
<img src="https://render.githubusercontent.com/render/math?math=\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)"><br>
where *(x)* is the indicator function.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Precision, recall and F-measures
The precision is the ratio **tp / (tp + fp)** where tp is the number of true positives and fp the number of false positives. The precision is intuitively the **ability of the classifier not to label as positive a sample that is negative**.<br><br>
The best value is 1 and the worst value is 0.

The **recall** is the ratio **tp / (tp + fn)** where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the **ability of the classifier to find all the positive samples**.

The best value is 1 and the worst value is 0.

The **F-measure** (<img src="https://render.githubusercontent.com/render/math?math=F_\beta"> and <img src="https://render.githubusercontent.com/render/math?math=F_1"> measures) can be interpreted as a weighted harmonic mean of the precision and recall. A <img src="https://render.githubusercontent.com/render/math?math=F_\beta"> measure reaches its best value at 1 and its worst score at 0. With <img src="https://render.githubusercontent.com/render/math?math=\beta = 1">, <img src="https://render.githubusercontent.com/render/math?math=F_\beta"> and <img src="https://render.githubusercontent.com/render/math?math=F_1"> are equivalent, and the recall and the precision are equally important.

The **average_precision_score** function computes the average precision (AP) from prediction scores. The value is between 0 and 1 and higher is better. AP is defined as: <br>
<img src="https://render.githubusercontent.com/render/math?math=\text{AP} = \sum_n (R_n - R_{n-1}) P_n"><br>
where <img src="https://render.githubusercontent.com/render/math?math=P_n"> and <img src="https://render.githubusercontent.com/render/math?math=R_n"> are the precision and recall at the nth threshold. With random predictions, the AP is the fraction of positive samples.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


To show how these metrics can evaluate our model, we first import the dataset for multiclass classification. We will be using the fruit dataset for classification of fruits on the basis of the features such as mass, height, etc. The dataframe can be downloaded from the following link:<br> https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/fruit_data_with_colors.txt


```python
fruits = pd.read_table(r'dataset/Fruit.txt')
fruits.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fruit_label</th>
      <th>fruit_name</th>
      <th>fruit_subtype</th>
      <th>mass</th>
      <th>width</th>
      <th>height</th>
      <th>color_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>apple</td>
      <td>granny_smith</td>
      <td>192</td>
      <td>8.4</td>
      <td>7.3</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>apple</td>
      <td>granny_smith</td>
      <td>180</td>
      <td>8.0</td>
      <td>6.8</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>apple</td>
      <td>granny_smith</td>
      <td>176</td>
      <td>7.4</td>
      <td>7.2</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>mandarin</td>
      <td>mandarin</td>
      <td>86</td>
      <td>6.2</td>
      <td>4.7</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>mandarin</td>
      <td>mandarin</td>
      <td>84</td>
      <td>6.0</td>
      <td>4.6</td>
      <td>0.79</td>
    </tr>
  </tbody>
</table>
</div>



We have 59 pieces of fruits, 7 features in the dataset and 4 types of fruits in the dataset:


```python
#Specify Targets and features
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

#split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Scaling Our data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(y_test.shape)
print(X_test.shape)
```

    (15,)
    (15, 4)
    


```python
# Import RF Regressor
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

#fit the model
model = rf.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


The methods that we have explained above have been demonstrated in the block below


```python
from sklearn import metrics
from sklearn.metrics import accuracy_score
print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))
print("Precision Score: ", metrics.precision_score(y_test, y_pred, average='macro'))
print("Recall_score: ", metrics.recall_score(y_test, y_pred, average='micro'))
print("f1_score: ", metrics.f1_score(y_test, y_pred, average='weighted'))
print("fbeta_score: ", metrics.fbeta_score(y_test, y_pred, average='macro', beta=0.5))
print("precision_recall_fscore_support: ", metrics.precision_recall_fscore_support(y_test, y_pred, beta=0.5, average=None))
```

    accuracy: 86.67%
    Precision Score:  0.8666666666666666
    Recall_score:  0.8666666666666667
    f1_score:  0.8675132275132273
    fbeta_score:  0.8712797619047619
    precision_recall_fscore_support:  (array([0.8       , 1.        , 1.        , 0.66666667]), array([1.  , 1.  , 0.75, 1.  ]), array([0.83333333, 1.        , 0.9375    , 0.71428571]), array([4, 1, 8, 2], dtype=int64))
    

## Confusion Matrix for MultiClass
The multilabel_confusion_matrix function **computes class-wise (default) or sample-wise (samplewise=True) multilabel confusion matrix to evaluate the accuracy of a classification**. multilabel_confusion_matrix also treats multiclass data as if it were multilabel, as this is a transformation commonly applied to evaluate multiclass problems with binary classification metrics (such as precision, recall, etc.).

When calculating class-wise multilabel confusion matrix *C*, the count of true negatives for class *i* is <img src="https://render.githubusercontent.com/render/math?math=C_{i,0,0}">, false negatives is <img src="https://render.githubusercontent.com/render/math?math=C_{i,1,0}">, true positives is <img src="https://render.githubusercontent.com/render/math?math=C_{i,1,1}"> and false positives is <img src="https://render.githubusercontent.com/render/math?math=C_{i,0,1}">


```python
from sklearn.metrics import multilabel_confusion_matrix
multilabel_confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None)
                 ]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.GnBu,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
plt.show()

```

    Confusion matrix, without normalization
    [[4 0 0 0]
     [0 1 0 0]
     [1 0 6 1]
     [0 0 0 2]]
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_98_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>



## ROC and AUC
An example of Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.

**ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis**. This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.

The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.

**ROC curves are typically used in binary classification** to study the output of a classifier. In order to extend ROC curve and ROC area to multi-label classification, **it is necessary to binarize the output**. **One ROC curve can be drawn per label**, but one can also draw a ROC curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging).

**Another evaluation measure for multi-label classification is macro-averaging, which gives equal weight to the classification of each label**.


```python
# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data just check which format is in the file before executing this code
fruits = pd.read_table(r'dataset/Fruit.txt')
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
# Binarize the output
Y_1 = label_binarize(y, classes=[1, 2, 3, 4])
n_classes = Y_1.shape[1]

# Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y_1, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state= 0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
```

The plot of a ROC curve for a specific class has been shown below



```python
plt.figure(figsize=(8,8))
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.text(0.41,0.8,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_103_0.png)


Plot ROC curves for all the classes ie a multilabel problem is shown below. It Computes macro-average ROC curve and ROC area. The Defference in Macro-average and micro-average is explained after the plot

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>




```python
fruits.fruit_label.unique()
```




    array([1, 2, 3, 4], dtype=int64)




```python
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(8,8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='black', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='grey', linestyle=':', linewidth=4)

colors = cycle(['lightskyblue', 'darkorange', 'gold', 'yellowgreen'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="best")
plt.text(0.4,0.6,'More accurate area',fontsize = 12, rotation=45)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12, rotation=45)
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_107_0.png)


**Area under ROC for the multiclass problem**

The func:`sklearn.metrics.roc_auc_score` function can be used for
multi-class classification. The multi-class One-vs-One scheme compares every
unique pairwise combination of classes. In this section, we calculate the AUC
using the OvR and OvO schemes. We report a macro average, and a
prevalence-weighted average.




```python
y_prob = classifier.predict_proba(X_test)

macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                  average="macro")
weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                     average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                     average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
```

    One-vs-One ROC AUC scores:
    0.858890 (macro),
    0.797464 (weighted by prevalence)
    One-vs-Rest ROC AUC scores:
    0.858890 (macro),
    0.797464 (weighted by prevalence)
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>



## Precision Recall Curve for Multiclass
The precision recall curve with the fruit dataset is shown below<br>
Precision-recall curves are typically used in binary classification to study the output of a classifier. In order to extend the precision-recall curve and average precision to multi-class or multi-label classification, it is necessary to binarize the output. One curve can be drawn per label, but one can also draw a precision-recall curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging).




```python
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

# Import some data 
fruits = pd.read_table(r'dataset/Fruit.txt')
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
random_state = np.random.RandomState(0)



# Limit to the two first classes, and split into training and test
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)
# Use label_binarize to be multi-label like settings
Y = label_binarize(y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
y_pred = classifier.predict(X_test)
```

The average precision score in multi-label settings


```python
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))
```

    Average precision score, micro-averaged over all classes: 0.34
    

Plot the micro-averaged Precision-Recall curve


```python
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
```




    Text(0.5, 1.0, 'Average precision score, micro-averaged over all classes: AP=0.34')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/5.%20Post-Modelling/output_116_1.png)

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>
