# Pre Processing

# Notebook Content
[Overview](#Data-Preprocessing)<br>
# Missing Value Treatments
[Missing Value Treatment with mean](#Mean-or-median-or-other-summary-statistic-substitution)<br>
[Forward and Backward fill](#Forward-fill-and-backward-fill)<br>
[Nearest neighbors imputation](#Nearest-neighbors-imputation)<br>
[MultiOutput Regressor](#MultiOutput-Regressor)<br>
[Iterative Imputer](#Iterative-Imputer)<br>
[Time-Series Specific Methods](#Time-Series-Specific-Methods)<br>
# Rescaling
[MinMaxScaler](#MinMaxScaler)<br>
[MaxAbsScaler](#MaxAbsScaler)<br>
[Robust Scaler](#Robust-Scaler)<br>
[StandardScaler](#StandardScaler)<br>
# Data Transformation
[Quantile Transformation](#Quantile-Transformation)<br>
[Power Transformation](#Power-Transformation)<br>
[Custom Transformation](#Custom-Transformation)<br>
[Data Normalization](#Data-Normalization)<br>
# Handling Categorical Variable
[One Hot Encoding](#One-Hot-Encoding)<br>
[Label Encoding](#Label-Encoding)<br>
[Hashing](#Hashing)<br>
[Backward Difference Encoding](#Backward-Difference-Encoding)<br>
# Embedding
[CountVectorizer](#CountVectorizer)<br>
[DictVectorizer](#DictVectorizer)<br>
[TF-IDF Vectorizer](#TF-IDF-Vectorizer)<br>
[Stemming](#Stemming)<br>
[Lemmatization](#Lemmatization)<br>
[Word2Vec](#Word2Vec)<br>
[Doc2Vec Embedding](#Doc2Vec-Embedding)<br>
[Visualize Word Embedding](#Visualize-Word-Embedding)<br>

# Data Preprocessing

Pre-processing refers to the **transformations applied to data** before feeding it to the algorithm. Data Preprocessing is a process that can be used to **convert the raw data into a clean dataset**. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for analysis; pre-processing heps us to bring our data to a desired format.

## Need for Data Preprocessing

**For achieving better results** from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning models need information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set. Another aspect is that data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithms are executed in one data set, and best out of them is chosen.

## Different data preprocesses

The different pre processing techniques are listed below; we will look into each of it in detail:

![Types%20of%20Pre_Processing.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/2.Pre-Processing/Data_Preprocess.png)

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# __Missing Value Treatments__
The methods used to handle missing values are as follows:<br>
1. Drop missing values
2. Fill missing value with test statistic
3. Predict missing value with maching learning algoritm


```python
import pandas as pd
import  numpy as np
# Check missing values in a dataset 
dict = {'First Score':[100, 90, np.nan, 95, 75], 

        'Second Score': [30, 45, 56, np.nan, np.nan], 

        'Third Score':[np.nan, 40, 98, 98, 56]} 

# creating a dataframe from list 
df = pd.DataFrame(dict)
print(df)
print('\nNo of null values:')
df.isnull().sum()
```

       First Score  Second Score  Third Score
    0        100.0          30.0          NaN
    1         90.0          45.0         40.0
    2          NaN          56.0         98.0
    3         95.0           NaN         98.0
    4         75.0           NaN         56.0
    
    No of null values:
    




    First Score     1
    Second Score    2
    Third Score     1
    dtype: int64




```python
# If the missing value isn’t identified as NaN , then we have to first convert or replace such non NaN entry with a NaN
df_2 = df.copy()
df_2['First Score'].replace(np.nan,0, inplace= True)
df_2[df_2['First Score'] == 0].head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>First Score</th>
      <th>Second Score</th>
      <th>Third Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>56.0</td>
      <td>98.0</td>
    </tr>
  </tbody>
</table>
</div>



## Imputation vs Removing Data
Before jumping to the methods of data imputation, we have to understand the reason why data goes missing.
1. **Missing completely at random**: This is a case when the probability of missing variable is same for all observations. For example: respondents of data collection process decide that they will declare their earning after tossing a fair coin. If an head occurs, respondent declares his / her earnings & vice versa. Here each observation has equal chance of missing value.
2. **Missing at random**: This is a case when variable is missing at random and missing ratio varies for different values / level of other input variables. For example: We are collecting data for age and female has higher missing value compare to male.
3. **Missing that depends on unobserved predictors**: This is a case when the missing values are not random and are related to the unobserved input variable. For example: In a medical study, if a particular diagnostic causes discomfort, then there is higher chance of drop out from the study. This missing value is not at random unless we have included “discomfort” as an input variable for all patients.
4. **Missing that depends on the missing value itself**: This is a case when the probability of missing value is directly correlated with missing value itself. For example: People with higher or lower income are likely to provide non-response to their earning.
 

**Simple approaches**<br>
A number of simple approaches exist. For basic use cases, these are often enough.<br><br>
**Dropping rows with null values**
1. If the number of data points is sufficiently high that dropping some of them will not cause lose generalizability in the models built (to determine whether or not this is the case, a learning curve can be used)
2. Dropping too much data is also dangerous
3. If in a large data set is present and missinng values is in range of 5-3%; then droping missing values is feasible


```python
df_3=df.copy()
df_3.dropna()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>First Score</th>
      <th>Second Score</th>
      <th>Third Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>90.0</td>
      <td>45.0</td>
      <td>40.0</td>
    </tr>
  </tbody>
</table>
</div>



**Dropping features with high nullity**

A feature that has a high number of empty values is unlikely to be very useful for prediction. It can often be safely dropped.
<br>**Note:** "But before deciding the variable is not usefull we should perform feature importance test for validation", tree based method can be used 


```python
df_2.drop(['Second Score'], axis= 1, inplace = True)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Mean or median or other summary statistic substitution
When to use example:
1. Check outlier, if less outliers is present then mean imputation can be used
2. When outliers are more median impuation can be used 
3. For categorical variables mode imputaion can be used

<br>**NOTE:**- Ok to use if missing data is less than 3%, otherwise introduces too much bias and artificially lowers variability of data


```python
# Simple illustration for missing value imputation with mean 
# The imputation strategies are mean, mode & median 
import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
SimpleImputer()
#This will look for all columns where we have NaN value and replace the NaN value with specified test statistic.
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))
```

    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]
    


```python
import pandas as pd
import numpy as np

df=pd.DataFrame([["XXL", 8, "black", "class 1", 22],
["L", np.nan, "gray", "class 2", 20],
["XL", 10, "blue", "class 2", 19],
["M", np.nan, "orange","class 1", 17],
["M", 11, "green", "class 3", np.nan],
["M", 7, "red", "class 1", 22]])

df.columns=["size", "price", "color", "class", "boh"]
df_copy= df.copy()
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>NaN</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>NaN</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# imputation is done with respect to one column by using mean, mode and median stragey 
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(df[['boh']])
df["boh"]=imp_mean.transform(df[["boh"]])
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>NaN</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>NaN</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# some other column 
imp_mean.fit(df[['price']])
df["price"]=imp_mean.transform(df[["price"]])
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>9.0</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>9.0</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Columns specific imputation in a dataframe 
df_4 = df.copy()
mean_value=df_4['price'].mean()
df_4['First score']=df_4['price'].fillna(mean_value)
#this will replace all NaN values with the mean of the non null values
#For Median
median_value=df_4['price'].median()
df_4['Second Score']=df_4['price'].fillna(median_value)
print(df_4)
```

      size  price   color    class   boh  First score  Second Score
    0  XXL    8.0   black  class 1  22.0          8.0           8.0
    1    L    9.0    gray  class 2  20.0          9.0           9.0
    2   XL   10.0    blue  class 2  19.0         10.0          10.0
    3    M    9.0  orange  class 1  17.0          9.0           9.0
    4    M   11.0   green  class 3  20.0         11.0          11.0
    5    M    7.0     red  class 1  22.0          7.0           7.0
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Forward fill and backward fill
Forward filling means fill missing values with previous data. Backward filling means fill missing values with next data point.


```python
# Creating the Series 
sr = pd.Series([100, None, None, 18, 65, None, 32, 10, 5, 24, 60]) 

# Create the Index 
index_ = pd.date_range('2010-10-09', periods = 11, freq ='M')   

# set the index
sr.index = index_   

# Print the series
print('Series  :\n',sr) 

```

    Series  :
     2010-10-31    100.0
    2010-11-30      NaN
    2010-12-31      NaN
    2011-01-31     18.0
    2011-02-28     65.0
    2011-03-31      NaN
    2011-04-30     32.0
    2011-05-31     10.0
    2011-06-30      5.0
    2011-07-31     24.0
    2011-08-31     60.0
    Freq: M, dtype: float64
    


```python
result = sr.fillna(method = 'ffill')
print('Series after forward fill :\n',result)

```

    Series after forward fill :
     2010-10-31    100.0
    2010-11-30    100.0
    2010-12-31    100.0
    2011-01-31     18.0
    2011-02-28     65.0
    2011-03-31     65.0
    2011-04-30     32.0
    2011-05-31     10.0
    2011-06-30      5.0
    2011-07-31     24.0
    2011-08-31     60.0
    Freq: M, dtype: float64
    


```python
result = sr.fillna(method = 'bfill')
print('Series after backward fill :\n',result)

```

    Series after backward fill :
     2010-10-31    100.0
    2010-11-30     18.0
    2010-12-31     18.0
    2011-01-31     18.0
    2011-02-28     65.0
    2011-03-31     32.0
    2011-04-30     32.0
    2011-05-31     10.0
    2011-06-30      5.0
    2011-07-31     24.0
    2011-08-31     60.0
    Freq: M, dtype: float64
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Nearest neighbors imputation
It can be used for data that are continuous, discrete, ordinal and categorical which makes it particularly useful for dealing with all kind of missing data. The assumption behind using KNN for missing values is that a point value can be approximated by the values of the points that are closest to it, based on other variables. <br><br>The distance metric varies according to the type of data:
1. **Continuous Data**: The commonly used distance metrics for continuous data are Euclidean, Manhattan and Cosine
2. **Categorical Data**: Hamming distance is generally used in this case. It takes all the categorical attributes 


```python
import numpy as np
from sklearn.impute import KNNImputer
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(X)
```




    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])




```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>9.0</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>9.0</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# KNN Imputer for a dataframe 
import numpy as np
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
q = imputer.fit_transform(df[['boh']])
df['boh'] = q
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>9.0</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>9.0</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
q = imputer.fit_transform(df[['price']])
df['price'] = q
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>9.0</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>9.0</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### MultiOutput Regressor


This strategy consists of fitting one regressor per target. This is a simple strategy for extending regressors that do not natively support multi-target regression.


```python
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
X, y = load_linnerud(return_X_y=True)
print(X)
print(y)
clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
pred =clf.predict(X[[0]])
```

    [[  5. 162.  60.]
     [  2. 110.  60.]
     [ 12. 101. 101.]
     [ 12. 105.  37.]
     [ 13. 155.  58.]
     [  4. 101.  42.]
     [  8. 101.  38.]
     [  6. 125.  40.]
     [ 15. 200.  40.]
     [ 17. 251. 250.]
     [ 17. 120.  38.]
     [ 13. 210. 115.]
     [ 14. 215. 105.]
     [  1.  50.  50.]
     [  6.  70.  31.]
     [ 12. 210. 120.]
     [  4.  60.  25.]
     [ 11. 230.  80.]
     [ 15. 225.  73.]
     [  2. 110.  43.]]
    [[191.  36.  50.]
     [189.  37.  52.]
     [193.  38.  58.]
     [162.  35.  62.]
     [189.  35.  46.]
     [182.  36.  56.]
     [211.  38.  56.]
     [167.  34.  60.]
     [176.  31.  74.]
     [154.  33.  56.]
     [169.  34.  50.]
     [166.  33.  52.]
     [154.  34.  64.]
     [247.  46.  50.]
     [193.  36.  46.]
     [202.  37.  62.]
     [176.  37.  54.]
     [157.  32.  52.]
     [156.  33.  54.]
     [138.  33.  68.]]
    


```python
pred
```




    array([[176.16484296,  35.0548407 ,  57.09000136]])



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Iterative Imputer

It is a Multivariate imputer that estimates each feature from all the others. It applies a strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.


```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
IterativeImputer(random_state=0)
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
imp_mean.transform(X)
```




    array([[ 6.95847623,  2.        ,  3.        ],
           [ 4.        ,  2.6000004 ,  6.        ],
           [10.        ,  4.99999933,  9.        ]])



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Time-Series Specific Methods
1. **Last Observation Carried Forward (LOCF) & Next Observation Carried Backward (NOCB)**
<br>This is a common statistical approach to the analysis of longitudinal repeated measured data where some follow-up observations may be missing. Longitudinal data track the same sample at different points in time. Both these methods can introduce bias in analysis and perform poorly when data has a visible trend
2. **Data without trend and seasonality**
mean, mode, median and random sample imputation can be used 
3. **Linear Interpolation**
This method works well for a time series with some **trend** but is not suitable for **seasonal data**
4. **Seasonal Adjustment + Linear Interpolation**
This method works well for data with both **trend and seasonality**


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Rescaling Data

When data is comprised of **attributes with varying scales**, many machine learning algorithms can benefit from rescaling the attributes to all have the same scale. This is useful for optimization algorithms used in the core of machine learning algorithms like gradient descent.

It is also useful for algorithms that weight inputs like regression and neural networks and algorithms that use distance measures like K-Nearest Neighbors. 
Rescaling of data using different techniques, some of which are listed below.

When faced with features which are very different in scale / units, it is quite clear to see that classifiers / regressors which rely on euclidean distance such as k-nearest neighbours will fail or be sub-optimal. Same goes for other regressors. Especially the ones that rely on gradient descent based optimisation such as logistic regressions, Support Vector Machines and Neural networks. The only classifiers/regressors which are immune to impact of scale are the tree based regressors.

**NOTE:** 
1. Before performing scalling one should check oultier and Treat the outlier
2. Check the EDA Notebook for various outlier treament method 

### MinMaxScaler

Transform features by **scaling each feature to a given range**.
This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.<br><br>The transformation is given by:<br>X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))<br>X_scaled = X_std * (max - min) + min


```python
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
MinMaxScaler()
print(scaler.transform(data))
```

    MinMaxScaler()
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>9.0</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>9.0</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# minmax scaler on cloumn of a dataframe
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['price'] = scaler.fit_transform(df[['price']])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>0.25</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>0.50</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>0.75</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.50</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>1.00</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>0.00</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>0.25</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>0.50</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>0.75</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.50</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>1.00</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>0.00</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

###  MaxAbsScaler

This estimator **scales and translates each feature individually** such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.<br><br>
This scaler can also be applied to sparse CSR or CSC matrices.


```python
from sklearn.preprocessing import MaxAbsScaler
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
transformer = MaxAbsScaler().fit(X)
transformer
MaxAbsScaler()
transformer.transform(X)

```




    array([[ 0.5, -1. ,  1. ],
           [ 1. ,  0. ,  0. ],
           [ 0. ,  1. , -0.5]])




```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>0.25</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>0.50</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>0.75</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.50</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>1.00</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>0.00</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(df[['price']])
df['price'] = transformer.transform(df[['price']])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>0.25</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>0.50</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>0.75</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.50</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>1.00</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>0.00</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Robust Scaler

Scale features using statistics that are **robust to outliers**. RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value)<br><br>
**Centering and scaling happen independently on each feature** by computing the relevant statistics on the samples in the training set. Median and interquartile range are then stored to be used on later data using the transform method.<br>
Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results<br><br>**Use RobustScaler, to reduce the effects of outliers**, relative to MinMaxScaler.


```python
from sklearn.preprocessing import RobustScaler
X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
transformer = RobustScaler().fit(X)
transformer
RobustScaler()
transformer.transform(X)

```




    array([[ 0. , -2. ,  0. ],
           [-1. ,  0. ,  0.4],
           [ 1. ,  0. , -1.6]])




```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>0.25</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>0.50</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>0.75</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.50</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>1.00</td>
      <td>green</td>
      <td>class 3</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>0.00</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# robustscaler for dataframe
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(df[['boh']])
df['boh'] = transformer.transform(df[['boh']])
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>0.25</td>
      <td>black</td>
      <td>class 1</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>0.50</td>
      <td>gray</td>
      <td>class 2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>0.75</td>
      <td>blue</td>
      <td>class 2</td>
      <td>-0.444444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>0.50</td>
      <td>orange</td>
      <td>class 1</td>
      <td>-1.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>1.00</td>
      <td>green</td>
      <td>class 3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>0.00</td>
      <td>red</td>
      <td>class 1</td>
      <td>0.888889</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### StandardScaler

StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance. Unit variance means dividing all the values by the standard deviation. StandardScaler does not meet the strict definition of scale I introduced earlier.

**When to use**
it can be used when to transform a feature so it is close to normally distributed 
**NOTE**
1. Results in the distribution with a Standard deviation equal to 1
2. If there are outliers in the feature, normalize the data and scale most of the data to a small interval



```python
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler


data= [[1, 1, 1, 1, 1],[2, 5, 10, 50, 100],[3, 10, 20, 150, 200],[4, 15, 40, 200, 300]]

df = pd.DataFrame(data, columns=['N0', 'N1', 'N2', 'N3', 'N4']).astype('float64')

sc_X = StandardScaler()
df = sc_X.fit_transform(df)

# df = pd.DataFrame(df, columns=['N0', 'N1', 'N2', 'N3', 'N4'])
# Get the dataframe for further analysis



# From this stats infromation can be obtanined
num_cols = len(df[0,:])
for i in range(num_cols):
    col = df[:,i]
    col_stats = ss.describe(col)
    print(col_stats)
```

    DescribeResult(nobs=4, minmax=(-1.3416407864998738, 1.3416407864998738), mean=0.0, variance=1.3333333333333333, skewness=0.0, kurtosis=-1.3599999999999999)
    DescribeResult(nobs=4, minmax=(-1.2828087129930659, 1.3778315806221817), mean=-5.551115123125783e-17, variance=1.3333333333333337, skewness=0.11003776770595125, kurtosis=-1.394993095506219)
    DescribeResult(nobs=4, minmax=(-1.155344148338584, 1.53471088361394), mean=0.0, variance=1.3333333333333333, skewness=0.48089217736510326, kurtosis=-1.1471008824318165)
    DescribeResult(nobs=4, minmax=(-1.2604572012883055, 1.2668071116222517), mean=-5.551115123125783e-17, variance=1.3333333333333333, skewness=0.0056842140599118185, kurtosis=-1.6438177182479734)
    DescribeResult(nobs=4, minmax=(-1.338945389819976, 1.3434309690153527), mean=5.551115123125783e-17, variance=1.3333333333333333, skewness=0.005374558840039456, kurtosis=-1.3619131970819205)
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Data Transformation

Two types of transformations are available: quantile transforms and power transforms.<br> 

###  Quantile Transformation
Quantile transformation can be used for __uniform data__. By performing a rank transformation, a quantile transform smooths out unusual distributions and is less influenced by outliers than scaling methods. It does, however, distort correlations and distances within and across features. <br><br> An example of Quantile Transformation is given below: 


```python
import numpy as np
from sklearn.preprocessing import QuantileTransformer
rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
qt = QuantileTransformer(n_quantiles=10, random_state=0)
qt.fit_transform(X)
```




    array([[0.        ],
           [0.09871873],
           [0.10643612],
           [0.11754671],
           [0.21017437],
           [0.21945445],
           [0.23498666],
           [0.32443642],
           [0.33333333],
           [0.41360794],
           [0.42339464],
           [0.46257841],
           [0.47112236],
           [0.49834237],
           [0.59986536],
           [0.63390302],
           [0.66666667],
           [0.68873101],
           [0.69611125],
           [0.81280699],
           [0.82160354],
           [0.88126439],
           [0.90516028],
           [0.99319435],
           [1.        ]])




```python
df
```




    array([[-1.34164079, -1.28280871, -1.15534415, -1.2604572 , -1.33894539],
           [-0.4472136 , -0.52262577, -0.53456222, -0.63816599, -0.45080071],
           [ 0.4472136 ,  0.4276029 ,  0.15519548,  0.63181608,  0.44631513],
           [ 1.34164079,  1.37783158,  1.53471088,  1.26680711,  1.34343097]])



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Power Transformation
Power transforms are a family of parametric transformations that aim to __map data from any distribution to as close to a Gaussian distribution__ as possible in order to stabilize variance and minimize skewness<br><br>There are two methods for power transformation: __Yeo Johnson and Box-cox__<br><br> Box-Cox can only be applied to strictly positive data. In both methods, the transformation is parameterized by 
λ, which is determined through maximum likelihood estimation. Here is an example of using Box-Cox to map samples drawn from a lognormal distribution to a normal distribution:


```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
pt.fit_transform(X_lognormal)

```




    array([[ 0.49024349,  0.17881995, -0.1563781 ],
           [-0.05102892,  0.58863195, -0.57612414],
           [ 0.69420009, -0.84857822,  0.10051454]])




```python
df_copy
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>NaN</td>
      <td>gray</td>
      <td>class 2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>NaN</td>
      <td>orange</td>
      <td>class 1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox', standardize=False)
df_copy['boh'] = pt.fit_transform(df_copy[['boh']])
df_copy
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>7546.664382</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>NaN</td>
      <td>gray</td>
      <td>class 2</td>
      <td>5524.660719</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>4670.996737</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>NaN</td>
      <td>orange</td>
      <td>class 1</td>
      <td>3245.914188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>7546.664382</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Custom Transformation
We might want to __convert an existing Python function into a transformer__ to assist in data cleaning or processing. A transformer from an arbitrary function with FunctionTransformer can be implemented. <br><br>For example, to build a transformer that applies a log transformation in a pipeline, the following can be done:


```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)
```




    array([[0.        , 0.69314718],
           [1.09861229, 1.38629436]])




```python
df_copy
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>7546.664382</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L</td>
      <td>NaN</td>
      <td>gray</td>
      <td>class 2</td>
      <td>5524.660719</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>4670.996737</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>NaN</td>
      <td>orange</td>
      <td>class 1</td>
      <td>3245.914188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>11.0</td>
      <td>green</td>
      <td>class 3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>7546.664382</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)
df_copy = df_copy.dropna()
df_copy['boh'] = transformer.fit_transform(df_copy[['boh']])
df_copy
```

    C:\Users\duhita\anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>8.928993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>8.449342</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>8.928993</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Data Normalization

Normalization is the process of **scaling individual samples to have unit norm**. The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms. Normalizer __works on the rows, not the columns!__ 
<br><br>By default, L2 normalization is applied to each observation so the that the values in a row have a unit norm. Unit norm with L2 means that if each element were squared and summed, the total would equal 1. Alternatively, L1 (aka taxicab or Manhattan) normalization can be applied instead of L2 normalization.


```python
from sklearn import preprocessing
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')
X_normalized

```




    array([[ 0.40824829, -0.40824829,  0.81649658],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678]])



The preprocessing module further provides a utility class Normalizer that implements the same operation using the Transformer API (even though the fit method is useless in this case: the class is stateless as this operation treats samples independently)


```python
normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer.transform(X)
```




    array([[ 0.40824829, -0.40824829,  0.81649658],
           [ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678]])




```python
df_copy
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>8.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>8.928993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>10.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>8.449342</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>7.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>8.928993</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import preprocessing
df_copy['price'] = preprocessing.normalize(df_copy[['price']], norm='l1')
df_copy

```

    C:\Users\duhita\anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>price</th>
      <th>color</th>
      <th>class</th>
      <th>boh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XXL</td>
      <td>1.0</td>
      <td>black</td>
      <td>class 1</td>
      <td>8.928993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XL</td>
      <td>1.0</td>
      <td>blue</td>
      <td>class 2</td>
      <td>8.449342</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M</td>
      <td>1.0</td>
      <td>red</td>
      <td>class 1</td>
      <td>8.928993</td>
    </tr>
  </tbody>
</table>
</div>



Normalize and Normalizer accept both dense array-like and sparse matrices from scipy.sparse as input. For sparse input the data is converted to the Compressed Sparse Rows representation (see scipy.sparse.csr_matrix) before being fed to efficient Cython routines. To avoid unnecessary memory copies, it is recommended to choose the CSR representation upstream.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### __To summarize__
• Use MinMaxScaler as the default if we are transforming a feature. It’s non-distorting.
<br>• Use RobustScaler if we have outliers and want to reduce their influence. However, we might be better off removing the outliers, instead.
<br>• Use StandardScaler if one need a relatively normal distribution.
<br>• Use Normalizer sparingly — it normalizes sample rows, not feature columns. It can use l2 or l1 normalization.

![Scalers.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/2.Pre-Processing/summary_normalization.png)

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# __Handling Categorical Variable__

### One Hot Encoding
In this method, each category is maped to a vector that contains 1 and 0 denoting the presence or absence of the feature. The number of vectors depends on the number of categories for features.<br><br>This method produces a lot of columns that slows down the learning significantly if the number of the category is very high for the feature.<br><br>One Hot Encoding is very popular. All categories can be represented by **N-1 (N= No of Category)** as that is sufficient to encode the one that is not included. Usually, for **Regression, N-1** (drop first or last column of One Hot Coded new feature ) is used, **but for classification, the recommendation is to use all N columns without as most of the tree-based algorithm builds a tree based on all available variables.**


```python
my_data = np.array([[5, 'a', 1],
                    [3, 'b', 3],
                    [1, 'b', 2],
                    [3, 'a', 1],
                    [4, 'b', 2],
                    [7, 'c', 1],
                    [7, 'c', 1]])                

df = pd.DataFrame(data=my_data, columns=['y', 'dummy', 'x'])
df = pd.get_dummies(df, columns = ['dummy'])
df
# Dummy variable are created 

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>x</th>
      <th>dummy_a</th>
      <th>dummy_b</th>
      <th>dummy_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>


```python
my_data = np.array([[5, 'a', 1],
                    [3, 'b', 3],
                    [1, 'b', 2],
                    [3, 'a', 1],
                    [4, 'b', 2],
                    [7, 'c', 1],
                    [7, 'c', 1]])                


df = pd.DataFrame(data=my_data, columns=['y', 'dummy', 'x'])
df = pd.get_dummies(df, columns = ['dummy'], drop_first = True)
# to run the regression we want to get rid of the strings 'a', 'b', 'c' (obviously)
# and we want to get rid of one dummy variable to avoid the dummy variable trap
# arbitrarily chose "a", coefficients on "c" an "b" would show effect of "c" and "b"
# relative to "a"
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>x</th>
      <th>dummy_b</th>
      <th>dummy_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Label Encoding
In this encoding, __each category is assigned a value from 1 through N__; here N is the number of categories for the feature. One major issue with this approach is that there is no relation or order between these classes, but the algorithm might consider them as some order, or there is some relationship.


```python
my_data = np.array([[5, 'a', 1],
                    [3, 'b', 3],
                    [1, 'b', 2],
                    [3, 'a', 1],
                    [4, 'b', 2],
                    [7, 'c', 1],
                    [7, 'c', 1]])                

df = pd.DataFrame(data=my_data, columns=['y', 'dummy', 'x'])
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>dummy</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>b</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>b</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>c</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>c</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['dummy'] = le.fit_transform(df.dummy)
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>dummy</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Hashing
Hashing converts categorical variables to a higher dimensional space of integers, where the distance between two vectors of categorical variables in approximately maintained the transformed numerical dimensional space.
<br><br>With Hashing, the __number of dimensions will be far less__ than the number of dimensions with encoding like One Hot Encoding. This method is **advantageous when the cardinality of categorical is very high**.

In Feature Hashing, a vector of categorical variables gets converted to a higher dimensional space of integers, where the distance between two vectors of categorical variables in approximately maintained the transformed numerical dimensional space. With Feature Hashing, the number of dimensions will be far less than the number of dimensions with simple binary encoding a.k.a One Hot Encoding.

Let’s consider the case of a data set with 2 categorical variables, the first one with a cardinality of 70 and the second one with a cardinality of 50. With simple binary encoding you will have to introduce 118 (70 + 50 – 2) additional fields to replace the 2 categorical variable fields in the data set.

With One Hot Encoding, the distance between categorical variables in any pair of records in preserved in the new space of dimension 118. With Feature Hashing you can get away with much smaller dimension e.g 10 in this case while recognizing that inter record distances will not be fully preserved. Hash collision is the reason for the failure to preserve the distances, making the mapping less than perfect.



```python
from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=10)
D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
f = h.transform(D)
f.toarray()

```




    array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
           [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Backward Difference Encoding
In backward difference coding, the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable.<br><br>This technique falls under the contrast coding system for categorical features. A feature of K categories, or levels, usually enters a regression as a sequence of K-1 dummy variables.


```python
# !pip install category_encoders
import category_encoders as ce
# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder(cols= ['dummy'],)
encoder.fit(df)

# Only display the first 8 columns for brevity
encoder.transform(df,override_return_df = False )
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>y</th>
      <th>dummy_0</th>
      <th>dummy_1</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>-0.666667</td>
      <td>-0.333333</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>0.333333</td>
      <td>-0.333333</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0.333333</td>
      <td>-0.333333</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>-0.666667</td>
      <td>-0.333333</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>0.333333</td>
      <td>-0.333333</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>7</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>7</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Embedding


**Text feature extraction**
<br>Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:

**a. tokenizing:** strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators

**b. counting:** the occurrences of tokens in each document

**c. normalizing:** weighting with diminishing importance tokens that occur in the majority of samples / documents

### CountVectorizer
The most straightforward one, it counts the number of times a token shows up in the document and uses this value as its weight.


```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
```

    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    


```python
X.toarray()
```




    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]], dtype=int64)



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### DictVectorizer
**Transforms lists of feature-value mappings to vectors**. This transformer turns lists of mappings (dict-like objects) of feature names to feature values into Numpy arrays or scipy.sparse matrices for use with scikit-learn estimators. <br><br>When feature values are strings, this transformer will do a binary one-hot (aka one-of-K) coding: one boolean-valued feature is constructed for each of the possible string values that the feature can take on. For instance, a feature “f” that can take on the values “ham” and “spam” will become two features in the output, one signifying “f=ham”, the other “f=spam”.<br>
<br>However, this transformer will only do a binary one-hot encoding when feature values are of type string. If categorical features are represented as numeric values such as int, the DictVectorizer can be followed by sklearn.preprocessing.OneHotEncoder to complete binary one-hot encoding.


```python
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)
X
```




    array([[2., 0., 1.],
           [0., 1., 3.]])



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### TF-IDF Vectorizer
It will **transform the text into the feature vectors** and used as input to the estimator. The vocabulary is the dictionary that will convert each token or word in the matrix and it will get the feature index. In **CountVectorizer** we only count the number of times a word appears in the document which results in biasing in favour of most frequent words. this ends up in ignoring rare words which could have helped is in processing our data more efficiently. To overcome this , we use TfidfVectorizer. <br>
<br>In **TfidfVectorizer** overall document weightage of a word is considered. It helps us in dealing with most frequent words.  TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',     'Is this the first document?']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
```

    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    


```python
# since X is not interpretable hence it ned to be transformed into dataframe with tf-idf score 
y = vectorizer.get_feature_names()
dense = X.todense()
dense_list = dense.tolist()
df = pd.DataFrame(dense_list, columns = y)
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>and</th>
      <th>document</th>
      <th>first</th>
      <th>is</th>
      <th>one</th>
      <th>second</th>
      <th>the</th>
      <th>third</th>
      <th>this</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.469791</td>
      <td>0.580286</td>
      <td>0.384085</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.384085</td>
      <td>0.000000</td>
      <td>0.384085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.687624</td>
      <td>0.000000</td>
      <td>0.281089</td>
      <td>0.000000</td>
      <td>0.538648</td>
      <td>0.281089</td>
      <td>0.000000</td>
      <td>0.281089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.511849</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.267104</td>
      <td>0.511849</td>
      <td>0.000000</td>
      <td>0.267104</td>
      <td>0.511849</td>
      <td>0.267104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.469791</td>
      <td>0.580286</td>
      <td>0.384085</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.384085</td>
      <td>0.000000</td>
      <td>0.384085</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.shape, df.shape
```




    ((4, 9), (4, 9))



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Stemming
Stemming is a kind of normalization for words. **Normalization** is a technique where a set of words in a sentence are converted into a sequence to shorten its lookup. The words which have the same meaning but have some variation according to the context or sentence are normalized. In another word, there is one root word, but there are many variations of the same words 
For example, the root word is "eat" and it's variations are "eats, eating, eaten and like so". In the same way, with the help of Stemming, we can find the root word of any variations. <br>
<br>NLTK has an algorithm named as "PorterStemmer". This algorithm accepts the list of tokenized word and stems it into root word 


```python
import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Stemming for {} is {}".format(w,porter_stemmer.stem(w))) 
```

    Stemming for studies is studi
    Stemming for studying is studi
    Stemming for cries is cri
    Stemming for cry is cri
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Lemmatization
Lemmatization is the algorithmic process of **finding the lemma of a word depending on their meaning**. Lemmatization usually refers to the morphological analysis of words, which aims to remove inflectional endings. It helps in returning the base or dictionary form of a word, which is known as the lemma. <br><br>
The NLTK Lemmatization method is based on WorldNet's built-in morph function. Text preprocessing includes both stemming as well as lemmatization. 

__Lemmatization is preferred over the former because of the below reason:__
Stemming algorithm works by cutting the suffix from the word. 
In a broader sense cuts either the beginning or end of the word. 
On the contrary, Lemmatization is a more powerful operation, and it takes into consideration morphological analysis 
of the words. It returns the lemma which is the base form of all its inflectional forms. 
In-depth linguistic knowledge is required to create dictionaries and look for the proper form of the word. 
Stemming is a general operation while lemmatization is an intelligent operation where the proper form will be 
looked in the dictionary. Hence, lemmatization helps in forming better machine learning features.


```python
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))  
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\duhita\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    

    Lemma for studies is study
    Lemma for studying is studying
    Lemma for cries is cry
    Lemma for cry is cry
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Word2Vec

Word2Vec consists of models for generating word embedding. These models are shallow two layer neural networks having one input layer, one hidden layer and one output layer. Word2Vec utilizes two architectures : <br>
__a. CBOW (Continuous Bag of Words) :__ CBOW model predicts the current word given context words within specific window. The input layer contains the context words and the output layer contains the current word. The hidden layer contains the number of dimensions in which we want to represent current word present at the output layer.<br>__b. Skip Gram :__ Skip gram predicts the surrounding context words within specific window given current word. The input layer contains the current word and the output layer contains the context words. The hidden layer contains the number of dimensions in which we want to represent current word present at the input layer.

![CBOWSKIP.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/dataset/Word2Vec.png)


```python
#install nltk and genism
#!pip install nltk
# !pip install gensim
```


```python
from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

```

    Word2Vec(vocab=14, size=100, alpha=0.025)
    ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec', 'second', 'yet', 'another', 'one', 'more', 'and', 'final']
    [-5.66251867e-04  3.87575617e-03  1.84540357e-03  1.58382370e-03
      9.45792941e-04  3.48803424e-03 -2.53669987e-03  4.93331812e-04
     -5.66694885e-04 -4.55752993e-03 -1.68438279e-03  1.13758014e-03
     -3.33453924e-03  8.24440503e-04  5.63966285e-04 -4.04788647e-03
      1.69406948e-03 -4.05336311e-03  2.18737638e-03  1.77856506e-04
      2.78158677e-05  2.15033023e-03  4.62671462e-03  4.71825758e-03
     -4.11912287e-03 -1.34883891e-03 -4.89438977e-03 -3.03815072e-03
      4.99660289e-03  3.80420941e-03  2.24203314e-03  4.65277862e-03
     -1.00864237e-03  2.10219412e-03 -2.07883582e-04  6.71489630e-04
     -3.19324154e-03 -1.05077273e-03  1.78553781e-03 -3.41403787e-03
      2.49195169e-03  1.33298442e-03  4.86205128e-04  4.76346258e-03
     -7.09757034e-04 -4.02095541e-03 -3.35455616e-03 -1.32341648e-03
     -2.16815024e-04 -4.70665609e-03 -2.04724167e-03 -1.16777934e-04
      4.37439606e-03 -1.03751954e-03  2.24079820e-03 -6.91010384e-04
     -4.70257038e-03  4.55049425e-03  4.05346742e-03 -4.42059152e-03
      1.40122499e-03 -3.44393332e-03 -3.14304419e-03 -4.65175416e-03
      2.62680417e-03 -1.88594521e-03  1.22818921e-03  3.29235499e-03
      4.15108050e-04 -1.94761634e-03 -2.65760138e-03  1.33196171e-03
      4.59288107e-03  1.28979026e-03  3.08361719e-04  1.59187324e-03
      3.68013163e-03  3.48706334e-03  3.36206541e-03 -4.19608271e-03
      1.25716499e-03 -4.62887064e-03 -2.98801158e-03  3.09391529e-03
     -1.59774232e-03 -4.93173162e-03 -2.87606660e-03  4.59354324e-03
      4.78505902e-03 -4.50887857e-03 -2.12219544e-03  2.98424345e-03
      4.89522424e-03 -4.83554974e-03 -4.55671968e-03 -4.59387852e-03
     -2.97720777e-03  1.62616151e-03 -4.12167748e-04 -2.81320349e-03]
    Word2Vec(vocab=14, size=100, alpha=0.025)
    

    C:\Users\duhita\anaconda3\lib\site-packages\ipykernel_launcher.py:16: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      app.launch_new_instance()
    

<br>• Output indicates the cosine similarities between word vectors ‘alice’, ‘wonderland’ and ‘machines’ for different models<br>• Both have their own advantages and disadvantages. Skip Gram works well with small amount of data and is found to represent rare words well.<br>
• On the other hand, CBOW is faster and has better representations for more frequent words.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Doc2Vec Embedding

__Syntax__ <br> class gensim.models.doc2vec.Doc2Vec(documents=None, corpus_file=None, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, callbacks=(), **kwargs)
<br> __How it works__ <br>• Doc2Vec is another widely used technique that creates an embedding of a document irrespective to its length. While Word2Vec computes a feature vector for every word in the corpus, Doc2Vec computes a feature vector for every document in the corpus<br>• Doc2vec model is based on Word2Vec, with only adding another vector (paragraph ID) to the input<br>• The Doc2Vec model, by analogy with Word2Vec, can rely on one of two model architectures which are: Distributed Memory version of Paragraph Vector (PV-DM) and Distributed Bag of Words version of Paragraph Vector (PV-DBOW)
<br>• In the figure below, we show the model architecture of PV-DM:

![Doc2Vec.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/dataset/Doc2Vec.png)

<br>• The above diagram is based on the CBOW model, but instead of using just nearby words to predict the word, we also added another feature vector, which is document-unique<br>• So when training the word vectors W, the document vector D is trained as well, and at the end of training, it holds a numeric representation of the document<br>• 
The inputs consist of word vectors and document Id vectors. The word vector is a one-hot vector with a dimension 1xV. The document Id vector has a dimension of 1xC, where C is the number of total documents. The dimension of the weight matrix W of the hidden layer is NxV. The dimension of the weight matrix D of the hidden layer is CxN


```python
#INITIALIZE AND TRAIN A MODEL
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

print("Corpus")
print('======')
print (common_texts) 
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

print('List of documents:')
print('==================')
for doc in documents:
    print(doc)

model = Doc2Vec(documents, size=5, window=2, min_count=1, workers=4)

#PERSIST A MODEL TO DESC:
 
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("my_doc2vec_model")
model.save(fname)
model = Doc2Vec.load(fname)  

#If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True) 

#INFER VECTOR FOR NEW DOCUMENT:
#Here our text paragraph just 2 words
vector = model.infer_vector(["system", "response"])
print ('\nVector format for words ==>',vector)
```

    Corpus
    ======
    [['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]
    List of documents:
    ==================
    TaggedDocument(['human', 'interface', 'computer'], [0])
    TaggedDocument(['survey', 'user', 'computer', 'system', 'response', 'time'], [1])
    TaggedDocument(['eps', 'user', 'interface', 'system'], [2])
    TaggedDocument(['system', 'human', 'system', 'eps'], [3])
    TaggedDocument(['user', 'response', 'time'], [4])
    TaggedDocument(['trees'], [5])
    TaggedDocument(['graph', 'trees'], [6])
    TaggedDocument(['graph', 'minors', 'trees'], [7])
    TaggedDocument(['graph', 'minors', 'survey'], [8])
    
    Vector format for words ==> [-0.01764038 -0.01116892 -0.08235094 -0.05593868 -0.03623259]
    

    C:\Users\duhita\anaconda3\lib\site-packages\gensim\models\doc2vec.py:319: UserWarning: The parameter `size` is deprecated, will be removed in 4.0.0, use `vector_size` instead.
      warnings.warn("The parameter `size` is deprecated, will be removed in 4.0.0, use `vector_size` instead.")
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Visualize Word Embedding
After we do word embedding for our text data, it can be nice to explore it with visualization.
We can use classical projection methods to reduce the high-dimensional word vectors to two-dimensional plots and plot them on a graph.
The visualizations can provide a qualitative diagnostic for our learned model.
We can retrieve all of the vectors from a trained model

We can then train a projection method on the vectors, such as those methods offered in scikit-learn, then use matplotlib to plot the projection as a scatter plot.

**Plot Word Vectors Using PCA**<br>
A 2-dimensional PCA model of the word vectors using the scikit-learn PCA class can be created as follows. The resulting projection can be plotted using matplotlib as follows, pulling out the two dimensions as x and y coordinates. We can go one step further and annotate the points on the graph with the words themselves. A crude version without any nice offsets looks as follows.


```python
#Putting this all together with the model from the previous section, the complete example is listed below.
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
['this', 'is', 'the', 'second', 'sentence'],
['yet', 'another', 'sentence'],
['one', 'more', 'sentence'],
['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

    C:\Users\duhita\anaconda3\lib\site-packages\ipykernel_launcher.py:14: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/2.Pre-Processing/output_154_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>
