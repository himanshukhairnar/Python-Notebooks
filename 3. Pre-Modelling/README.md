# Pre Modeling

# Notebook Content
[Library](#Load-the-requred-Libraries-and-Dataset)<br>
# Sampling
[Sampling](#Sampling)<br>
[Simple Random Sampling](#Simple-Random-Sampling)<br>
[Stratified-Sampling](#Stratified-Sampling)<br>
[Random Undersampling and Oversampling](#Random-Undersampling-and-Oversampling)<br>
[Undersampling](#Undersampling)<br>
[Oversampling](#Oversampling)<br>
[Reservoir Sampling](#Reservoir-Sampling)<br>
[Undersampling and Oversampling using imbalanced learn](#Undersampling-and-Oversampling-using-imbalanced-learn)<br>
[ClusterCentroids](#ClusterCentroids)<br>
[Undersampling using Tomek-Links](#Undersampling-using-Tomek-Links)<br>
[Oversampling Methods](#Oversampling-Methods)<br>
[SMOTE](#SMOTE)<br>
[ADASYN](#ADASYN)<br>
[Borderline Smote](#Borderline-Smote)<br>
[SVM SMOTE](#SVM-SMOTE)<br>
[KMeans SMOTE](#KMeans-SMOTE)<br>
# Feature importance
[Feature importance](#1)<br>
[Univariate Selection](#Univariate-Selection)<br>
[Recursive Feature Elimination](#Recursive-Feature-Elimination)<br>
[Removing Highly Corelated Variables](#Removing-Highly-Corelated-Variables)<br>
[Boruta](#Boruta)<br>
[Variance Inflation Factor (VIF)](#Variance-Inflation-Factor)<br>
[Principal Component Analysis (PCA)](#Principal-Component-Analysis)<br>
[Linear Discriminant Analysis (LDA)](#Linear-Discriminant-Analysis)<br>
[Feature Importance using RF](#Feature-Importance-using-RF)<br>


## Load the requred Libraries and Dataset


```python
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.formula.api import ols

```

The download links for the automobile dataset that have been used in the notebook is given below. Please download the dataset before proceeding
https://www.kaggle.com/toramky/automobile-dataset
    


```python
auto_mobile = pd.read_csv(r'dataset/auto_mobile.csv')
```

# Sampling
Data sampling refers to **statistical methods for selecting observations from the domain** with the objective of estimating a population parameter. Whereas data resampling refers to methods for economically using a collected dataset to improve the estimate of the population parameter and help to quantify the uncertainty of the estimate.

### Simple Random Sampling


Simple Random Sampling can be used to select a subset of a population in which each member of the subset has an equal probability of being chosen.
Below is a code to select 100 sample points from a dataset.


```python
sample_df = auto_mobile.sample(100)
auto_mobile.shape, sample_df.shape
```




    ((203, 27), (100, 27))



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Stratified Sampling
This technique divides the elements of the population into small subgroups (strata) based on the similarity in such a way that the **elements within the group are homogeneous and heterogeneous among the other subgroups formed**. And then the elements are randomly selected from each of these strata. Prior information about the population is required to create subgroups.


```python
x = auto_mobile.drop(columns = 'price', axis = 1)
y = auto_mobile['price']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)
X_train.shape, X_test.shape
```




    ((162, 26), (41, 26))



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Random Undersampling and Oversampling

**Imbalanced data:** A dataset is imbalanced if at least one of the classes constitutes only a very small minority. Imbalanced data prevail in banking, insurance, engineering, and many other fields. It is common in fraud detection that the imbalance is on the order of 100 to 1

The issue of class imbalance can result in a serious bias towards the majority class, reducing the classification performance and increasing the number of false negatives. How can we alleviate the issue? The most commonly used techniques are data resampling either under-sampling the majority of the class, or oversampling the minority class, or a mix of both. This will result in improved classification performance.

**The Problem with Imbalanced Classes**
Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce error.

![Under_over_Sample.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/dataset/Undersampling%20and%20oversampling.png)

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


**It is too often that we encounter an imbalanced dataset**<br>
A widely adopted technique for dealing with highly imbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and/or adding more examples from the minority class (over-sampling).


```python
# here is the small illustration of imbalance dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)
X = pd.DataFrame(X)
X['target'] = y
print(x.shape)
X.head()
```

    (203, 26)
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.327419</td>
      <td>-0.123936</td>
      <td>0.377707</td>
      <td>-0.650123</td>
      <td>0.267562</td>
      <td>1.228781</td>
      <td>2.208772</td>
      <td>-0.185977</td>
      <td>0.238732</td>
      <td>-2.565438</td>
      <td>...</td>
      <td>0.644056</td>
      <td>0.104375</td>
      <td>-1.703024</td>
      <td>-0.510083</td>
      <td>-0.108812</td>
      <td>-0.230132</td>
      <td>1.553707</td>
      <td>1.497538</td>
      <td>-1.476485</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.843981</td>
      <td>-0.018691</td>
      <td>-0.841018</td>
      <td>1.374583</td>
      <td>0.157199</td>
      <td>-0.599719</td>
      <td>2.217041</td>
      <td>-2.032194</td>
      <td>-2.310214</td>
      <td>-0.490477</td>
      <td>...</td>
      <td>1.360939</td>
      <td>-1.844740</td>
      <td>-0.341096</td>
      <td>0.137243</td>
      <td>1.704764</td>
      <td>0.464255</td>
      <td>1.225786</td>
      <td>-0.842880</td>
      <td>1.303258</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.204642</td>
      <td>0.472155</td>
      <td>-0.140616</td>
      <td>-2.902493</td>
      <td>-1.513665</td>
      <td>1.149545</td>
      <td>2.283673</td>
      <td>-0.809117</td>
      <td>-1.723535</td>
      <td>-0.958556</td>
      <td>...</td>
      <td>-0.279701</td>
      <td>-1.431391</td>
      <td>0.260146</td>
      <td>-0.501306</td>
      <td>-2.320545</td>
      <td>0.422214</td>
      <td>1.386474</td>
      <td>-0.073335</td>
      <td>0.586859</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.208274</td>
      <td>-0.156982</td>
      <td>0.063369</td>
      <td>-0.545759</td>
      <td>-0.395416</td>
      <td>-2.679969</td>
      <td>1.507772</td>
      <td>0.391485</td>
      <td>-0.487337</td>
      <td>-0.946147</td>
      <td>...</td>
      <td>-1.011854</td>
      <td>-1.124795</td>
      <td>0.347291</td>
      <td>-1.078836</td>
      <td>0.046923</td>
      <td>-0.978324</td>
      <td>1.100517</td>
      <td>-0.697134</td>
      <td>0.339577</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.785568</td>
      <td>0.208472</td>
      <td>0.760082</td>
      <td>-0.046130</td>
      <td>0.310844</td>
      <td>-0.403927</td>
      <td>1.462897</td>
      <td>0.962173</td>
      <td>-0.520996</td>
      <td>1.647360</td>
      <td>...</td>
      <td>0.316792</td>
      <td>-0.261528</td>
      <td>-1.260698</td>
      <td>0.822700</td>
      <td>0.141031</td>
      <td>-0.294805</td>
      <td>2.216364</td>
      <td>-1.129875</td>
      <td>-1.059984</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
num_0 = len(X[X['target']==0])
num_1 = len(X[X['target']==1])
print(num_0,num_1)

```

    90 10
    


```python
ax = X['target'].value_counts().plot(kind='bar',
                                    figsize=(5,5),
                                    title="Representation of Imbalance Data", color='lightblue', width=0.3)
ax.set_xlabel("Target")
ax.set_ylabel("Frequency")

# it can be observed how data is distributed 
# majority class "0"
# minority classs "1"
```




    Text(0, 0.5, 'Frequency')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_22_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Undersampling


```python
undersampled_data = pd.concat([ X[X['target']==0].sample(num_1) , X[X['target']==1] ])
print(len(undersampled_data))
undersampled_data.head()
```

    20
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0.008699</td>
      <td>1.624146</td>
      <td>0.520398</td>
      <td>-1.017329</td>
      <td>0.556007</td>
      <td>0.490563</td>
      <td>2.320411</td>
      <td>-0.002059</td>
      <td>-1.380078</td>
      <td>-1.433130</td>
      <td>...</td>
      <td>0.605618</td>
      <td>1.646173</td>
      <td>-0.099357</td>
      <td>0.454205</td>
      <td>-0.834171</td>
      <td>-0.929712</td>
      <td>1.386173</td>
      <td>-1.972335</td>
      <td>-0.528811</td>
      <td>0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>-0.196458</td>
      <td>-1.560088</td>
      <td>1.404330</td>
      <td>-0.872605</td>
      <td>-2.197019</td>
      <td>0.426422</td>
      <td>2.695514</td>
      <td>0.273345</td>
      <td>-1.956011</td>
      <td>-1.433979</td>
      <td>...</td>
      <td>-0.339199</td>
      <td>0.217618</td>
      <td>-0.304961</td>
      <td>0.098799</td>
      <td>0.723541</td>
      <td>-0.536805</td>
      <td>1.537721</td>
      <td>0.795029</td>
      <td>0.378958</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1.411594</td>
      <td>-0.235615</td>
      <td>-0.580188</td>
      <td>-1.600235</td>
      <td>-0.205932</td>
      <td>1.128180</td>
      <td>3.017396</td>
      <td>-0.479210</td>
      <td>-2.056289</td>
      <td>-1.859780</td>
      <td>...</td>
      <td>-0.021252</td>
      <td>-1.206703</td>
      <td>-0.005576</td>
      <td>-0.302466</td>
      <td>0.694198</td>
      <td>0.000560</td>
      <td>1.672228</td>
      <td>1.611499</td>
      <td>0.119738</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.607495</td>
      <td>-0.342030</td>
      <td>-2.864752</td>
      <td>0.648469</td>
      <td>1.182989</td>
      <td>-2.257192</td>
      <td>0.310184</td>
      <td>-0.248203</td>
      <td>-0.562329</td>
      <td>0.756881</td>
      <td>...</td>
      <td>-0.764580</td>
      <td>-0.143524</td>
      <td>-2.021147</td>
      <td>0.375832</td>
      <td>-0.090405</td>
      <td>0.684720</td>
      <td>0.428258</td>
      <td>0.524235</td>
      <td>-1.066375</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1.833188</td>
      <td>1.306245</td>
      <td>-1.123141</td>
      <td>-0.306433</td>
      <td>0.108924</td>
      <td>0.862164</td>
      <td>1.623093</td>
      <td>0.825020</td>
      <td>-0.046326</td>
      <td>-1.399979</td>
      <td>...</td>
      <td>-0.628833</td>
      <td>-0.945038</td>
      <td>-0.082833</td>
      <td>-0.760949</td>
      <td>0.720260</td>
      <td>0.314024</td>
      <td>1.251654</td>
      <td>-0.646506</td>
      <td>1.443162</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
ax = undersampled_data['target'].value_counts().plot(kind='bar',
                                    figsize=(5,5),
                                    title="Representation of Imbalance Data", color='yellowgreen', width=0.3)
ax.set_xlabel("Target")
ax.set_ylabel("Frequency")

# it can be observed that by using under sampling the data is converted into balance data
# minority class is 1, count of 1 was 10 in original dataset
# balance data contains equal proportion of target variable based on minority class
  
print(undersampled_data.shape)
```

    (20, 21)
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_26_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Oversampling


```python
oversampled_data = pd.concat([ X[X['target']==0] , X[X['target']==1].sample(num_0, replace=True) ])
print(len(oversampled_data))
oversampled_data.head()
```

    180
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.327419</td>
      <td>-0.123936</td>
      <td>0.377707</td>
      <td>-0.650123</td>
      <td>0.267562</td>
      <td>1.228781</td>
      <td>2.208772</td>
      <td>-0.185977</td>
      <td>0.238732</td>
      <td>-2.565438</td>
      <td>...</td>
      <td>0.644056</td>
      <td>0.104375</td>
      <td>-1.703024</td>
      <td>-0.510083</td>
      <td>-0.108812</td>
      <td>-0.230132</td>
      <td>1.553707</td>
      <td>1.497538</td>
      <td>-1.476485</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.843981</td>
      <td>-0.018691</td>
      <td>-0.841018</td>
      <td>1.374583</td>
      <td>0.157199</td>
      <td>-0.599719</td>
      <td>2.217041</td>
      <td>-2.032194</td>
      <td>-2.310214</td>
      <td>-0.490477</td>
      <td>...</td>
      <td>1.360939</td>
      <td>-1.844740</td>
      <td>-0.341096</td>
      <td>0.137243</td>
      <td>1.704764</td>
      <td>0.464255</td>
      <td>1.225786</td>
      <td>-0.842880</td>
      <td>1.303258</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.204642</td>
      <td>0.472155</td>
      <td>-0.140616</td>
      <td>-2.902493</td>
      <td>-1.513665</td>
      <td>1.149545</td>
      <td>2.283673</td>
      <td>-0.809117</td>
      <td>-1.723535</td>
      <td>-0.958556</td>
      <td>...</td>
      <td>-0.279701</td>
      <td>-1.431391</td>
      <td>0.260146</td>
      <td>-0.501306</td>
      <td>-2.320545</td>
      <td>0.422214</td>
      <td>1.386474</td>
      <td>-0.073335</td>
      <td>0.586859</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.208274</td>
      <td>-0.156982</td>
      <td>0.063369</td>
      <td>-0.545759</td>
      <td>-0.395416</td>
      <td>-2.679969</td>
      <td>1.507772</td>
      <td>0.391485</td>
      <td>-0.487337</td>
      <td>-0.946147</td>
      <td>...</td>
      <td>-1.011854</td>
      <td>-1.124795</td>
      <td>0.347291</td>
      <td>-1.078836</td>
      <td>0.046923</td>
      <td>-0.978324</td>
      <td>1.100517</td>
      <td>-0.697134</td>
      <td>0.339577</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.886195</td>
      <td>0.548814</td>
      <td>-1.844824</td>
      <td>0.638066</td>
      <td>0.023932</td>
      <td>0.491861</td>
      <td>0.722346</td>
      <td>0.811078</td>
      <td>-0.468527</td>
      <td>0.035382</td>
      <td>...</td>
      <td>-0.751144</td>
      <td>0.148616</td>
      <td>-0.185694</td>
      <td>2.102140</td>
      <td>-0.166839</td>
      <td>0.088302</td>
      <td>0.632036</td>
      <td>1.766467</td>
      <td>-1.373949</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
ax = oversampled_data['target'].value_counts().plot(kind='bar',
                                    figsize=(5,5),
                                    title="Representation of Imbalance Data", color='gold', width=0.3)
ax.set_xlabel("Target")
ax.set_ylabel("Frequency")

# it can be observed that by using over sampling the data is converted into balance data
# majority class is 0, count of 0 was 90 in original dataset  
# balance data contains equal proportion of target variable based on majority class
print(oversampled_data.shape)
```

    (180, 21)
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_30_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Reservoir Sampling
In this sampling each element has the same probability of getting selected from the stream. Let us assume we have to sample 5 objects out of an infinite stream such that each element has an equal probability of getting selected. An exapmle of reservoir sampling is given below:


```python
import random
def generator(max):
    number = 1
    while number < max:
        number += 1
        yield number
# Create as stream generator
stream = generator(10000)
# Doing Reservoir Sampling from the stream
k=5
reservoir = []
for i, element in enumerate(stream):
    if i+1<= k:
        reservoir.append(element)
    else:
        probability = k/(i+1)
        if random.random() < probability:
            # Select item in stream and remove one of the k items already selected
             reservoir[random.choice(range(0,k))] = element
print(reservoir)
```

    [3182, 8456, 1862, 5008, 9176]
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Undersampling and Oversampling using imbalanced-learn
Imbalanced-learn(imblearn) is a Python Package to tackle the curse of imbalanced datasets.
It provides a variety of methods to undersample and oversample.

The imblearn.under_sampling provides various methods to under-sample a dataset


## a. Prototype generation
The imblearn.under_sampling.prototype_generation submodule contains methods that generate new samples in order to balance the dataset.


### ClusterCentroids
Perform **under-sampling by generating centroids** based on clustering methods. Method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples.<br><br>
Supports multi-class resampling by sampling each class independently. An example for under sampling using Cluster Centroids is given below:


```python
#!pip install imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids 

X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))

cc = ClusterCentroids(random_state=42)
X_res, y_res = cc.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))


```

    Original dataset shape Counter({1: 900, 0: 100})
    Resampled dataset shape Counter({0: 100, 1: 100})
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## b. Prototype selection
The imblearn.under_sampling.prototype_selection submodule contains methods that select samples in order to balance the dataset.
The methods in Prototype Selection that can be used is listed below:<br>

|METHOD|DESCRIPTION|
|------|-----------|
|under_sampling.CondensedNearestNeighbour|Class to perform under-sampling based on the condensed nearest neighbour method|
|under_sampling.EditedNearestNeighbours|Class to perform under-sampling based on the edited nearest neighbour method|
|under_sampling.RepeatedEditedNearestNeighbours|Class to perform under-sampling based on the repeated edited nearest neighbour method|
|under_sampling.AllKNN|Class to perform under-sampling based on the AllKNN method|
|under_sampling.InstanceHardnessThreshold|Class to perform under-sampling based on the instance hardness threshold|
|under_sampling.NearMiss|Class to perform under-sampling based on NearMiss methods|
|under_sampling.NeighbourhoodCleaningRule|Class performing under-sampling based on the neighbourhood cleaning rule|
|under_sampling.OneSidedSelection|Class to perform under-sampling based on one-sided selection method|
|under_sampling.RandomUnderSampler|Class to perform random under-sampling|
|under_sampling.TomekLinks|Class to perform under-sampling by removing Tomek’s links|

### Undersampling using Tomek Links
The most comming under sampling method is the Tomek Links method and hence we will be showing it in detail<br>
Tomek links are pairs of examples of opposite classes in close vicinity<br>
In this algorithm, we end up removing the majority element from the Tomek link which provides a better decision boundary for a classifier<br>

![Tomeklinks.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/dataset/Tomeklinks.png)


```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks 

X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))
tl = TomekLinks()
X_res, y_res = tl.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
```

    Original dataset shape Counter({1: 900, 0: 100})
    Resampled dataset shape Counter({1: 897, 0: 100})
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Oversampling Methods
The imblearn.over_sampling provides a set of method to perform over-sampling
The different over sampling techniques that can be used are listed below:<br>

|METHOD|DESCRIPTION|
|------|-----------|
|over_sampling.ADASYN|Perform over-sampling using Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets|
|over_sampling.BorderlineSMOTE|Over-sampling using Borderline SMOTE|
|over_sampling.KMeansSMOTE|Apply a KMeans clustering before to over-sample using SMOTE|
|over_sampling.RandomOverSampler|Class to perform random over-sampling|
|over_sampling.SMOTE|Class to perform over-sampling using SMOTE|
|over_sampling.SMOTENC(categorical_features)|Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTE-NC)|
|over_sampling.SVMSMOTE|Over-sampling using SVM-SMOTE|

Apart from the random sampling with replacement, there are two popular methods to over-sample minority classes: <br>
(i) the Synthetic Minority Oversampling Technique (SMOTE) and <br>
(ii) the Adaptive Synthetic (ADASYN) sampling method<br>
These algorithms can be used in the same manner:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### SMOTE
In SMOTE (Synthetic Minority Oversampling Technique) we synthesize elements for the minority class, in the vicinity of already existing elements<br>
Supports multi-class resampling

![smote.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/dataset/SMOTE.png)


```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

```

    Original dataset shape Counter({1: 900, 0: 100})
    Resampled dataset shape Counter({0: 900, 1: 900})
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### ADASYN
Over-sampling can be performed using Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets.


```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000,
random_state=10)
print('Original dataset shape %s' % Counter(y))

ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

```

    Original dataset shape Counter({1: 900, 0: 100})
    Resampled dataset shape Counter({0: 904, 1: 900})
    

• While the RandomOverSampler is over-sampling by duplicating some of the original samples of the minority class, **SMOTE and ADASYN generate new samples in by interpolation**<br>
• The samples used to interpolate/generate new synthetic samples differ. In fact, ADASYN focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier while the basic implementation of SMOTE will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule <br>


**SMOTE might connect inliers and outliers while ADASYN might focus solely on outliers which, in both cases, might lead to a sub-optimal decision function**. In this regard, SMOTE offers three additional options to generate samples the codes for which are mentioned below:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Borderline Smote
Borderline samples will be detected and used to generate new synthetic samples. It supports multi-class resampling.


```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE # doctest: +NORMALIZE_WHITESPACE
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))

sm = BorderlineSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

```

    Original dataset shape Counter({1: 900, 0: 100})
    Resampled dataset shape Counter({0: 900, 1: 900})
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### SVM SMOTE
Uses an SVM algorithm to detect sample to use for generating new synthetic samples. It supports multi-class resampling


```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SVMSMOTE 
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))

sm = SVMSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

```

    Original dataset shape Counter({1: 900, 0: 100})
    Resampled dataset shape Counter({0: 900, 1: 900})
    

For both borderline and SVM SMOTE, a neighborhood is defined using the parameter m_neighbors to decide if a sample is in danger, safe, or noise.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### KMeans SMOTE
Uses a KMeans clustering method before to apply SMOTE. The clustering will group samples together and generate new samples depending of the cluster density.


```python
import numpy as np
from imblearn.over_sampling import KMeansSMOTE
from sklearn.datasets import make_blobs
blobs = [100, 800, 100]
X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
# Add a single 0 sample in the middle blob
X = np.concatenate([X, [[0, 0]]])
y = np.append(y, 0)
# Make this a binary classification problem
y = y == 1
sm = KMeansSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
# Find the number of new samples in the middle blob
n_res_in_middle = ((X_res[:, 0] > -5) & (X_res[:, 0] < 5)).sum()
print("Samples in the middle blob: %s" % n_res_in_middle)
print("Middle blob unchanged: %s" % (n_res_in_middle == blobs[1] + 1))
print("More 0 samples: %s" % ((y_res == 0).sum() > (y == 0).sum()))

```

    Samples in the middle blob: 801
    Middle blob unchanged: True
    More 0 samples: True
    

• All algorithms can be used with multiple classes as well as binary classes classification<br>
• RandomOverSampler does not require any inter-class information during the sample generation<br>
• Therefore, each targeted class is resampled independently. In the contrary, both ADASYN and SMOTE need information regarding the neighbourhood of each sample used for sample generation. They are using a one-vs-rest approach by selecting each targeted class and computing the necessary statistics against the rest of the data set which are grouped in a single class

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

<a id = '1'></a>
# Feature importance 
The performance of machine learning model is directly proportional to the data features used to train it. The performance of ML model will be affected negatively if the data features provided to it are irrelevant. On the other hand, **use of relevant data features can increase the accuracy of your ML model** especially linear and logistic regression.

Performing feature selection before data modeling will reduce the overfitting.<br>
Performing feature selection before data modeling will increases the accuracy of ML model.<br>
Performing feature selection before data modeling will reduce the training time.<br>

### Univariate Selection
This feature selection technique is very useful in **selecting relevent features, with the help of statistical testing**, having strongest relationship with the prediction variables. We can implement univariate feature selection technique with the help of **SelectKBest()** class of scikit-learn Python library.<br>
There are **two parameter for SelectKBest()**: <br>
1. First is the statistical testing function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. Default is f_classif (see below “See also”). The default function only works with classification tasks. Other fuctions available:<br>

|Fuction|Use|
|-------|---|
|f_classif|ANOVA F-value between label/feature for classification tasks|
|mutual_info_classif|Mutual information for a discrete target|
|chi2|Chi-squared stats of non-negative features for classification tasks|
|f_regression|F-value between label/feature for regression tasks|
|mutual_info_regression|Mutual information for a continuous target|
|SelectPercentile|Select features based on percentile of the highest scores|
|SelectFpr|Select features based on a false positive rate test|
|SelectFdr|Select features based on an estimated false discovery rate|
|SelectFwe|Select features based on family-wise error rate|
|GenericUnivariateSelect|Univariate feature selector with configurable mode|

2. Number of top features to select. The “all” option bypasses selection, for use in a parameter search.

In the example below, we will use Pima Indians Diabetes dataset to select 4 of the attributes having best features with the help of chi-square statistical test. Please download the dataset from the link below: <br>
https://www.kaggle.com/kumargh/pimaindiansdiabetescsv



```python
# example of chi squared feature selection
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
 
# load the dataset
def load_dataset(filename):
 # load the dataset as a pandas DataFrame
    data = read_csv(filename)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
 
# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k=4)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
# load the dataset
X, y = load_dataset(r'dataset/pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_, color='lightskyblue')
pyplot.show()
```

    Feature 0: 47.636124
    Feature 1: 859.808560
    Feature 2: 0.245516
    Feature 3: 18.481768
    Feature 4: 2036.862034
    Feature 5: 82.085045
    Feature 6: 3.089404
    Feature 7: 104.405213
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_73_1.png)


Now, in the above example we have done for a dataset where all the input variables are numerical. So below is an example of feature selection using SelecKBest() on a dataframe where there are categorical variables in the input. In such cases we have to do encoding before runing the algorithm. The dataset used below can be downloaded from the following link:<br>
https://github.com/datasets/breast-cancer/tree/master/data


```python
# example of chi squared feature selection for independent categorical data 
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
 
# load the dataset
def load_dataset(filename):
 # load the dataset as a pandas DataFrame
    data = read_csv(filename)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, 1:-1]
    y = dataset[:,-1]
    # format all fields as string
    X = X.astype(str)
    return X, y
 
# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
 
# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
# load the dataset
X, y = load_dataset('dataset/breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_, color='lightskyblue')
pyplot.show()
```

    Feature 0: 0.472553
    Feature 1: 0.029193
    Feature 2: 2.137658
    Feature 3: 29.381059
    Feature 4: 0.776891
    Feature 5: 8.100183
    Feature 6: 1.273822
    Feature 7: 0.125944
    Feature 8: 3.699989
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_75_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Recursive Feature Elimination
RFE (Recursive feature elimination) feature selection technique **removes the attributes recursively and builds the model with remaining attributes**. We can implement RFE feature selection technique with the help of RFE class of scikit-learn Python library.<br><br>
Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. <br>First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.<br><br>
In the example below, we will use RFE with logistic regression algorithm to select the best 3 attributes having the best features from Pima Indians Diabetes dataset to.<br>
Please download the dataset from the link below: <br>
https://www.kaggle.com/kumargh/pimaindiansdiabetescsv



```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
dataframe = pd.read_csv(r'dataset/pima-indians-diabetes.csv')
array = dataframe.values
```


```python
#Next, we will separate the array into its input and output components −
X = array[:,0:8]
Y = array[:,8]
```


```python
#The following lines of code will select the best features from a dataset −
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features: 3")
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)
```

    Number of Features: 3
    Selected Features:  [ True False False False False  True  True False]
    Feature Ranking:  [1 2 4 5 6 1 1 3]
    

In above output it can be seen, RFE choose Number of times pregnant, Body mass index and Diabetes pedigree function' as the first 3 best features. They are marked as 1 in the output.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Removing Highly Corelated Variables

Correlation is a statistical term which in common usage refers to how close two variables are to having a linear relationship with each other.
<br>For example, two variables which are linearly dependent (say, x and y which depend on each other as x = 2y) will have a higher correlation than two variables which are non-linearly dependent (say, u and v which depend on each other as u = v^2)

**How does correlation help in feature selection?**
<br>Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable(multicollinearity). So, when two features have high correlation, we can drop one of the two features<br>
We will be showing how it works using the data for breast cancer.<br> The link to download dataset:https://www.kaggle.com/uciml/breast-cancer-wisconsin-data


```python
df = pd.read_csv('dataset/cancer.csv')
df.drop(columns = 'Unnamed: 32', axis = 1, inplace = True)
df.shape
```




    (569, 32)




```python
X = df.drop(columns = ['diagnosis'], axis = 1)
print(X.shape)
Y = df['diagnosis']
a = df.corr()
plt.rcParams['figure.figsize']=(15,15)
ax = sns.heatmap(a, linewidth=0.5, cmap= 'BuGn_r', annot = True)
plt.show()

```

    (569, 31)
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_87_1.png)



```python
columns = np.full((a.shape[0],), True, dtype=bool)
print(len(columns))
for i in range(a.shape[0]):
    for j in range(i+1, a.shape[0]):
        if a.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
X = X[selected_columns]
```

    31
    


```python
# In can be seen from above initally contains 31 variable
# but after Removing highly correlated variables its shape changes to
X.shape
```




    (569, 21)



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Boruta
The Boruta algorithm is a wrapper built around the random forest classification algorithm. It tries to capture all the important, interesting features in the data set with respect to an outcome variable.

**Methodology:**
1. First it creates randomness to the features by creating duplicate features and shuffle the values in each column. This features are called Shadow Features.
2. Trains a classifier (Random Forest) on the Dataset and calculate the importance using Mean Decrease Accuracy or Mean Decrease Impurity.
3. Then, the algorithm checks for each of your real features if they have higher importance. That is, whether the feature has a higher Z-score than the maximum Z-score of its shadow features than the best of the shadow features.
4. At every iteration, the algorithm compares the Z-scores of the shuffled copies of the features and the original features to see if the latter performed better than the former. If it does, the algorithm will mark the feature as important.


```python
# !pip install boruta
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
```


```python
df.shape
```




    (569, 32)




```python
# Data Preprocessing 
SEED = 999
df = pd.read_csv('dataset/cancer.csv')
X, y = df.drop(['diagnosis','id','Unnamed: 32'], axis=1), df['diagnosis']      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


# Standardize data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)


# Feature Selection 
from boruta import BorutaPy

# Random Forests for Boruta 
rf_boruta = RandomForestClassifier(n_jobs=-1, random_state=SEED)
# Perform Boruta
boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=2)
boruta.fit(X_train.values, y_train.values.ravel())

# Select features 
cols = X_train.columns[boruta.support_]
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	2 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	3 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	4 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	5 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	6 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	7 / 100
    Confirmed: 	0
    Tentative: 	30
    Rejected: 	0
    Iteration: 	8 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	9 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	10 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	11 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	12 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	13 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	14 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	15 / 100
    Confirmed: 	20
    Tentative: 	10
    Rejected: 	0
    Iteration: 	16 / 100
    Confirmed: 	21
    Tentative: 	7
    Rejected: 	2
    Iteration: 	17 / 100
    Confirmed: 	21
    Tentative: 	7
    Rejected: 	2
    Iteration: 	18 / 100
    Confirmed: 	21
    Tentative: 	7
    Rejected: 	2
    Iteration: 	19 / 100
    Confirmed: 	21
    Tentative: 	7
    Rejected: 	2
    Iteration: 	20 / 100
    Confirmed: 	21
    Tentative: 	6
    Rejected: 	3
    Iteration: 	21 / 100
    Confirmed: 	21
    Tentative: 	6
    Rejected: 	3
    Iteration: 	22 / 100
    Confirmed: 	21
    Tentative: 	6
    Rejected: 	3
    Iteration: 	23 / 100
    Confirmed: 	21
    Tentative: 	6
    Rejected: 	3
    Iteration: 	24 / 100
    Confirmed: 	21
    Tentative: 	6
    Rejected: 	3
    Iteration: 	25 / 100
    Confirmed: 	21
    Tentative: 	6
    Rejected: 	3
    Iteration: 	26 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	27 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	28 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	29 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	30 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	31 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	32 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	33 / 100
    Confirmed: 	21
    Tentative: 	5
    Rejected: 	4
    Iteration: 	34 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	35 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	36 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	37 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	38 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	39 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	40 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	41 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	42 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	43 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	44 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	45 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	46 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	47 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	48 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	49 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	50 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	51 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	52 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	53 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	54 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	55 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	56 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	57 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	58 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	59 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	60 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	61 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	62 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	63 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	64 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	65 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	66 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	67 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	68 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	69 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	70 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	71 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	72 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	73 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	74 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	75 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	76 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	77 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	78 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	79 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	80 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	81 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	82 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	83 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	84 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	85 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	86 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	87 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	88 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	89 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	90 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	91 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	92 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	93 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	94 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	95 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	96 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	97 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	98 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    Iteration: 	99 / 100
    Confirmed: 	22
    Tentative: 	4
    Rejected: 	4
    
    
    BorutaPy finished running.
    
    Iteration: 	100 / 100
    Confirmed: 	22
    Tentative: 	2
    Rejected: 	4
    


```python
# Number of columns before applying feature engineering technique: Boruta
print(len(df.columns))
# Number of cloumns after applying Boruta
print(len(cols))
```

    33
    22
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Variance Inflation Factor

Variance Inflation Factor (VIF) is used to detect the presence of multicollinearity. Variance inflation factors (VIF) measure how much the variance of the estimated regression coefficients are inflated as compared to when the predictor variables are not linearly related.

It is obtained by regressing each independent variable, say X on the remaining independent variables (say Y and Z) and checking how much of it (of X) is explained by these variables.


  **VIF = 1/(1- R^2)**

**Application & Interpretation:**<br>
1. From the list of variables, we select the variables with high VIF as collinear variables. But to decide which variable to select, we look at the Condition Index of the variables or the final regression coefficient table.
2. As a thumb rule, any variable with VIF > 2 is avoided in a regression analysis. Sometimes the condition is relaxed to 5, instead of 2.
3. It is not suitable for categorical variables


```python
# same dataset is used as above
# Remove the target variable and other columns like say "id", "product_id" or unwanted vairable
df = df.drop(columns = ['id', 'diagnosis','Unnamed: 32'], axis = 1)
df.shape
```




    (569, 30)




```python
from sklearn.linear_model import LinearRegression
def Vif_cal(X):
    vif_check = X.copy()
    vif = []
    var = []
    for i in vif_check.columns:
        lr = LinearRegression()
        lr.fit(vif_check.drop([i],axis=1),vif_check[i])
        r2 = lr.score(vif_check.drop([i],axis=1),vif_check[i])
        vif.append(1/(1-r2))
        var.append(i)
    vif_df= pd.Series(vif, index=var)
    vif_df = pd.DataFrame({'variable':var,'vif':vif})
    vif_df.sort_values(by = 'vif', ascending=False,inplace=True)
    return(vif_df)
    return(vif)
    
```


```python
Vif_cal(df).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>vif</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>radius_mean</td>
      <td>3806.115296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>perimeter_mean</td>
      <td>3786.400419</td>
    </tr>
    <tr>
      <th>20</th>
      <td>radius_worst</td>
      <td>799.105946</td>
    </tr>
    <tr>
      <th>22</th>
      <td>perimeter_worst</td>
      <td>405.023336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>area_mean</td>
      <td>347.878657</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Principal Component Analysis
PCA, generally called **data reduction technique**, is very useful feature selection technique as it uses linear algebra to transform the dataset into a compressed form. We can implement PCA feature selection technique with the help of PCA class of scikit-learn Python library. We can select number of principal components in the output.

**The components are formed such that it explains the maximum variation in the dataset**.

In the example below, we will use PCA to select best 3 Principal components from Pima Indians Diabetes dataset.
PCA can be used were model interpritbilty is not concern. <br>Download link for diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv



```python
from sklearn.decomposition import PCA
dataframe = pd.read_csv(r'dataset/pima-indians-diabetes.csv')
array = dataframe.values
```


```python
#Next, we will separate array into input and output components −
X = array[:,0:8]
Y = array[:,8]
```


```python
#The following lines of code will extract features from dataset −
pca = PCA(n_components = 3)
fit = pca.fit(X)
print("Explained Variance: ", fit.explained_variance_ratio_)
print(fit.components_)
print(fit.n_features_)
print(fit.explained_variance_)
```

    Explained Variance:  [0.88854663 0.06159078 0.02579012]
    [[-2.02176587e-03  9.78115765e-02  1.60930503e-02  6.07566861e-02
       9.93110844e-01  1.40108085e-02  5.37167919e-04 -3.56474430e-03]
     [-2.26488861e-02 -9.72210040e-01 -1.41909330e-01  5.78614699e-02
       9.46266913e-02 -4.69729766e-02 -8.16804621e-04 -1.40168181e-01]
     [-2.24649003e-02  1.43428710e-01 -9.22467192e-01 -3.07013055e-01
       2.09773019e-02 -1.32444542e-01 -6.39983017e-04 -1.25454310e-01]]
    8
    [13456.57298102   932.76013231   390.57783115]
    

PCA does not reak pick features foe further modeling, it reduces large numer of features to the number of components we want. <br>
PCA does not provide much interpretability

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Linear Discriminant Analysis
LDA is a type of Linear combination, a mathematical process using various data items and applying a function to that site to separately analyze multiple classes of objects or items.



**In case of LDA, the transform method takes two parameters: the X_train and the y_train. However in the case of PCA, the transform method only requires one parameter i.e. X_train**. This reflects the fact that LDA takes the output class labels into account while selecting the linear discriminants, while PCA doesn't depend upon the output labels.

We have shown an example of LDA below:


```python
dataframe = pd.read_csv(r'dataset/pima-indians-diabetes.csv')
array = dataframe.values
```


```python
#Next, we will separate array into input and output components −
X = array[:,0:8]
Y = array[:,8]

```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
X_test.shape
```




    (154, 1)



• Here it can be seen that one components are passed, it is because LDA takes into consideration the number of classes in the dependent variables(output class labels), ie, the number of categories in y_train. <br>
• In the dataset the dependent is 'Class' which has either value '1', meaning a person has diabetes or '0' when the person does not have diabetes.<br> 
• For our description of LDA, the k−1 dimensions we keep would be the in-model space, whereas the remaining dimensions are the out-of-model space, k being the number of output class labels.<br>
• A visualization for how PCA and LDA works is given below:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


Principal Component Analysis (**PCA**) applied to this data **identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance** in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (**LDA**) tries to **identify attributes that
account for the most variance** between classes. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.




```python
print(__doc__)

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
# print(X)
# print(y)
print("target names:" , target_names)

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
```

    Automatically created module for IPython interactive environment
    target names: ['setosa' 'versicolor' 'virginica']
    explained variance ratio (first two components): [0.92461872 0.05306648]
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_122_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_122_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Feature Importance using RF
It uses a **trained supervised classifier to select features**. We can implement this feature selection technique with the help of any tree based classifier of scikit-learn Python library.<br>
It is a widely used method for feature selection. <br>

In the example below, RandomForestClassifier is used to select features from Pima Indians Diabetes dataset.


```python
#Feature importance using Randomforest Classifier (independent-numerical, target-categorical)
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
 
# load the dataset
def load_dataset(filename):
 # load the dataset as a pandas DataFrame
    data = read_csv(filename)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
 
# feature selection
def select_features(X_train, y_train, X_test):
    fs = RandomForestClassifier()
    fs.fit(X_train, y_train)
    X_train_fs = fs.apply(X_train)
    X_test_fs = fs.apply(X_test)
    return X_train_fs, X_test_fs, fs
 
# load the dataset
X, y = load_dataset(r'dataset/pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are feature importances for the features
for i in range(len(fs.feature_importances_)):
    print('Feature %d: %f' % (i, fs.feature_importances_[i]))
# plot the feature importance
pyplot.bar([i for i in range(len(fs.feature_importances_))], fs.feature_importances_, color='gold')
pyplot.show()
```

    Feature 0: 0.079879
    Feature 1: 0.234760
    Feature 2: 0.103441
    Feature 3: 0.067670
    Feature 4: 0.077548
    Feature 5: 0.176165
    Feature 6: 0.133996
    Feature 7: 0.126542
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/3.Pre-Modelling/output_125_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
