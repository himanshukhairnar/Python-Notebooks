# Exploratory Data Analysis (EDA)

# Notebook Content
[Overview](#Exploratory-Data-Analysis)<br>
[Library](#Load-the-required-Libraries-and-Dataset)<br>
[Data Cleaning](#Data-Cleaning)<br>
# Univariate Analysis
[Categorical Variables](#Categorical-Variables)<br>
[Pie Chart](#Pie-Chart)<br>
[Bar Plot](#Bar-Plot)<br>
[Frequency Table](#Frequency-Table)<br>
[Numerical Variables](#Numerical-Variables)<br>
[Histograms](#Histograms)<br>
[Box plot](#Boxplot)<br>
[Density Plot](#Density-Plot)<br>
# Bivariate analysis
[Numerical-Numerical](#Numerical-Numerical)<br>
[Scatter-plot](#Scatter-plot)<br>
[Categorical Numerical](#Categorical-Numerical)<br>
[Bar Plots](#Bar-Plots)<br>
[Box Plots](#Box-Plots)<br>
[Categorical Categorical](#Categorical-Categorical)<br>
[Cross Tab](#Cross-Tab)<br>
[Combination Chart](#Combination-Chart)<br>
[Z Test](#Z-Test)<br>
[T Test](#T-Test)<br>
[Correlation](#Correlation)<br>
[Chi Square Tests](#Chi-Square-Tests)<br>
# Multivariate Analysis
[Combination Chart](#Combination-Chart)<br>
[Scatter Plot](#Scatter-Plot)<br>
[Pair Plot](#Pair-Plot)<br>
# Outlier Treatment
[Z Score](#Z-Score)<br>
[IQR Score](#IQR-Score)<br>

# Exploratory Data Analysis
Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations. There are three types of EDA:<br>
1. Univariate Analysis<br>
2. Bivariate Analysis<br>
3. Multivariate Analysis<br>

Each of these will be explained in detail below:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Load the required Libraries and Dataset


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
from IPython.display import Image

```

The dataset can be downloaded from the link given below:<br>
https://www.kaggle.com/toramky/automobile-dataset


```python
#import dataset
auto_mobile = pd.read_csv('dataset/Automobile_data.csv')
auto_mobile.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>




```python
#Getting the data types of the dataset
auto_mobile.dtypes
```




    symboling              int64
    normalized-losses     object
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                  object
    stroke                object
    compression-ratio    float64
    horsepower            object
    peak-rpm              object
    city-mpg               int64
    highway-mpg            int64
    price                 object
    dtype: object




```python
#Getting the Statistics of the Data
auto_mobile.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>compression-ratio</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.834146</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>10.142537</td>
      <td>25.219512</td>
      <td>30.751220</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.245307</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>3.972040</td>
      <td>6.542142</td>
      <td>6.886443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>8.600000</td>
      <td>19.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>9.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>9.400000</td>
      <td>30.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>23.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Finding out if there are Null values
auto_mobile.isna().sum()
```




    symboling            0
    normalized-losses    0
    make                 0
    fuel-type            0
    aspiration           0
    num-of-doors         0
    body-style           0
    drive-wheels         0
    engine-location      0
    wheel-base           0
    length               0
    width                0
    height               0
    curb-weight          0
    engine-type          0
    num-of-cylinders     0
    engine-size          0
    fuel-system          0
    bore                 0
    stroke               0
    compression-ratio    0
    horsepower           0
    peak-rpm             0
    city-mpg             0
    highway-mpg          0
    price                0
    dtype: int64



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Data Cleaning

Dataset contains few '?' entries instead of real values. Hence,to clean the columns of the data:


```python
#Cleaning normalized-losses
# Find out number of records having '?' value for normalized losses
auto_mobile['normalized-losses'].loc[auto_mobile['normalized-losses'] == '?'].count()


# Setting the missing value to mean of normalized losses and conver the datatype to integer
nl = auto_mobile['normalized-losses'].loc[auto_mobile['normalized-losses'] != '?']
nlmean = nl.astype(str).astype(int).mean()
auto_mobile['normalized-losses'] = auto_mobile['normalized-losses'].replace('?',nlmean).astype(int)


#Cleaning price data: a. Checking if there are non numeric values
#Find out the number of values which are not numeric
auto_mobile['price'].str.isnumeric().value_counts()
# List out the values which are not numeric
auto_mobile['price'].loc[auto_mobile['price'].str.isnumeric() == False]


#Setting the missing value to mean of price and convert the datatype to integer
price = auto_mobile['price'].loc[auto_mobile['price'] != '?']
pmean = price.astype(str).astype(int).mean()
auto_mobile['price'] = auto_mobile['price'].replace('?',pmean).astype(int)
auto_mobile['price'].head()
```




    0    13495
    1    16500
    2    16500
    3    13950
    4    17450
    Name: price, dtype: int32




```python
#Cleaning the horsepower data: a. Checking if there are non numeric values
#                              b. Checking if any value is greater than 10000
#Checking the numberic and replacing with mean value and conver the datatype to integer
auto_mobile['horsepower'].str.isnumeric().value_counts()
horsepower = auto_mobile['horsepower'].loc[auto_mobile['horsepower'] != '?']
hpmean = horsepower.astype(str).astype(int).mean()
auto_mobile['horsepower'] = auto_mobile['horsepower'].replace('?',pmean).astype(int)


#Checking the outlier of horsepower
auto_mobile.loc[auto_mobile['horsepower'] > 10000]

#Excluding the outlier data for horsepower
auto_mobile[np.abs(auto_mobile.horsepower-auto_mobile.horsepower.mean())<=(3*auto_mobile.horsepower.std())]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.4</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.4</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>202</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>203</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.4</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>204</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>203 rows × 26 columns</p>
</div>




```python
#Cleaning the bore: a. Find if any values are invalid(?)
# Find out the number of invalid value
auto_mobile['bore'].loc[auto_mobile['bore'] == '?']


# Replace the non-numeric value to null and conver the datatype
auto_mobile['bore'] = pd.to_numeric(auto_mobile['bore'],errors='coerce')

#Cleaning the stroke
# Replace the non-number value to null and convert the datatype
auto_mobile['stroke'] = pd.to_numeric(auto_mobile['stroke'],errors='coerce')

#Checking the peak rpm data
# Convert the non-numeric data to null and convert the datatype
auto_mobile['peak-rpm'] = pd.to_numeric(auto_mobile['peak-rpm'],errors='coerce')


#Cleaning athe number of doors data
# remove the records which are having the value '?'
auto_mobile['num-of-doors'].loc[auto_mobile['num-of-doors'] == '?']
auto_mobile = auto_mobile[auto_mobile['num-of-doors'] != '?']
auto_mobile['num-of-doors'].loc[auto_mobile['num-of-doors'] == '?']
```




    Series([], Name: num-of-doors, dtype: object)



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

# UNIVARIATE ANALYSIS 

Univariate analysis explores variables (attributes) one by one. Variables could be either categorical or numerical. There are different statistical and visualization techniques of investigation for each type of variable. Numerical variables can be transformed into categorical counterparts by a process called binning or discretization. It is also possible to transform a categorical variable into its numerical counterpart by a process called encoding. <br>
<br>**Univariate Analysis** can be further classified in two categories :
1. Categorical
2. Numerical



### Categorical Variables
A categorical or discrete variable is one that has two or more categories (values). There are two types of categorical variables, nominal and ordinal.  A nominal variable has no intrinsic ordering to its categories. For example, gender is a categorical variable having two categories (male and female) with no intrinsic ordering to the categories. An ordinal variable has a clear ordering. For example, temperature as a variable with three orderly categories (low, medium and high).<br>A frequency table is a way of counting how often each category of the variable in question occurs. It may be enhanced by the addition of percentages that fall into each category:<br>

|Statistics|Visualization|Description|
|----------|-------------|-----------|
|Count|Bar Chart|The number of values of the specified variable|
|Count%|Pie Chart|The percentage of values of the specified variable|



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Pie Chart
A pie chart is a **circular statistical diagram**. The area of the whole chart represents 100% or the whole of the data. The **areas of the pies present in the Pie chart represent the percentage of parts of data**. The parts of a pie chart are called **wedges**. The length of the arc of a wedge determines the area of a wedge in a pie chart. The area of the wedges determines the relative quantum or percentage of a part with respect to a whole. Pie charts are frequently used in business presentations as they give quick summary of the business activities like sales, operations and so on. Pie charts are also used heavily in survey results, news articles, resource usage diagrams like disk and memory.


```python
#number of unique body-styles
auto_mobile['body-style'].unique()
```




    array(['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'],
          dtype=object)




```python
## pie chart can be made to find the percentage of each body-style
labels = list(auto_mobile['body-style'].unique())
sizes = list(auto_mobile['body-style'].value_counts())
colors = ['yellowgreen', 'lightskyblue', 'purple','coral','gold']
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice
fig = plt.figure(figsize = (8,8)) 

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.legend(labels, loc="best")

plt.axis('equal')
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_24_0.png)



```python
print(sizes)
```

    [94, 70, 25, 8, 6]
    

From the pie chart the percentage of each bode style of automobiles can be visualised.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

###  Bar Plot
A bar chart or bar graph is a chart or graph that **presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent**.<br>

The bars can be plotted vertically or horizontally.<br>

A bar graph shows comparisons among discrete categories. One axis of the chart shows the specific categories being compared, and the other axis represents a measured value.


```python
#make dataframe for plotting
d = auto_mobile.make.value_counts().to_frame().reset_index() 
  
# Set the index 
d.columns = ['Make','count']
d.head()


```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>toyota</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nissan</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mazda</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mitsubishi</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>honda</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# for vertical bar plot
def vertical_bar_plot(df, column_name, xlabel, ylabel,title):
    labels = d.Make.tolist()
    y_pos = labels
    sizes = df[column_name].tolist()
    fig = plt.figure(figsize = (8,5)) 
    #plot

    plt.bar(y_pos, sizes, align='center', alpha=1, width=0.4, color='yellowgreen')
    plt.xticks(y_pos, labels, rotation ='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.show()
    
# df -dataframe
# column_name - colummn for which plot needs to me made (it should passed as a string eg: 'count')
# xlabel-- variable on x-axis (it should be passed in a format -'xlabel')
# ylabel-- variable on y-axis (it should be passed in a format -'xlabel')
# title-- title of a chart it should be passed in string format 

vertical_bar_plot(d,'count','make','number','Number of each make')
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_31_0.png)



```python
#for horizontal bar plots
def horizontal_bar_plot(df, column, xlabel, ylabel,title):
    labels = df.Make.tolist()
    y_pos = labels
    sizes = df[column].tolist()
    fig = plt.figure(figsize = (8,6)) 
    #plot
    plt.barh(y_pos, sizes, align='center', alpha=1, height=0.4, color='gold')
    plt.yticks(y_pos, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.show()
    
# df -dataframe
# column_name - colummn for which plot needs to me made (it should passed as a string eg: 'count')
# xlabel-- variable on x-axis (it should be passed in a format -'xlabel')
# ylabel-- variable on y-axis (it should be passed in a format -'xlabel')
# title-- title of a chart it should be passed in string format 
horizontal_bar_plot(d,'count','make','number','Number of each make')
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_31_0.png)


From the two histograms a count of the number of cars present in each category of body style can be seen.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Frequency Table

Frequency table checks the **count of each category in a particular column of the data** and along with this percentage of data in each category can also be found out.


```python
#for frequency table
df = pd.DataFrame()
df = pd.value_counts(auto_mobile.make).to_frame()
total = df.make.sum()
df['percentage(%) of make']=(df['make']/total)*100 
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make</th>
      <th>percentage(%) of make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>toyota</th>
      <td>32</td>
      <td>15.763547</td>
    </tr>
    <tr>
      <th>nissan</th>
      <td>18</td>
      <td>8.866995</td>
    </tr>
    <tr>
      <th>mazda</th>
      <td>16</td>
      <td>7.881773</td>
    </tr>
    <tr>
      <th>mitsubishi</th>
      <td>13</td>
      <td>6.403941</td>
    </tr>
    <tr>
      <th>honda</th>
      <td>13</td>
      <td>6.403941</td>
    </tr>
  </tbody>
</table>
</div>



From the table above it can be said that toyota makes the highest and mercury makes the lowest number of cars. Similarly such tables can be made for all the categorical variables.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Numerical Variables
A numerical or continuous variable (attribute) is one that may **take on any value within a finite or infinite interval** (e.g., height, weight, temperature, blood glucose, ...).<br>

There are two types of numerical variables, interval and ratio. An interval variable has values whose differences are interpretable, but it does not have a true zero. A good example is temperature in Centigrade degrees. Data on an interval scale can be added and subtracted but cannot be meaningfully multiplied or divided. For example, we cannot say that one day is twice as hot as another day. In contrast, a ratio variable has values with a true zero and can be added, subtracted, multiplied or divided (e.g., weight).<br>

|Statistics|Visualization|Description|
|----------|-------------|-----------|
|Count|Histogram|The number of observations of the variable|
|Minimum|Box Plot|The smallest value of the variable|
|Maximum|Box Plot|The largest value of the variable|
|Mean|Box Plot|The sum of the values divided by the count|
|Median|Box Plot|The middle value|
|Mode|Histogram|The most frequent value|
|Quantile|Box Plot|A set of 'cut points' that divide a set of data into groups containing equal numbers of values|
|Range|Box Plot|The difference between maximum and minimum|
|Variance|Histogram|A measure of data dispersion|
|Standard Deviation|Histogram|The square root of variance|
|Coefficient of Deviation|Histogram|A measure of data dispersion divided by mean|
|Skewness|Histogram|A measure of symmetry or asymmetry in the distribution of data|
|Kurtosis|Histogram|A measure of whether the data are peaked or flat relative to a normal distribution|



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Histograms
The purpose of a histogram is to **graphically summarize the distribution of a univariate data set**.<br><br>
**The histogram graphically shows the following:** centre  (i.e., the location) of the data, spread (i.e., the scale) of the data, skewness of the data, presence of outliers, presence of multiple modes in the data.<br>
The most common form of the histogram is obtained by splitting the range of the data into equal-sized bins (called classes). Then for each bin, the number of points from the data set that fall into each bin are counted. That is;<br>
Vertical axis: Frequency (i.e., counts for each bin) <br>
Horizontal axis: Response variable <br>

_The histogram can be used to answer the following questions:_ <br>
1. What kind of population distribution do the data come from<br>
2. Where are the data located<br>
3. How spread out are the data<br>
4. Are the data symmetric or skewed<br> 
5. Are there outliers in the data<br>
The code to plot a histogram has been given below:


```python
# function for histogram
def hist_plot(column_name, title):
    l = auto_mobile[column_name]
    fig = plt.figure(figsize = (8,8)) 
    #adding subplots
    ax1 = fig.add_subplot(221)

    #plot
    #plot of length
    ax1.hist(l,bins=5, histtype='bar', align='mid', alpha= 1, color='lightskyblue', label='length data', edgecolor='black')
    ax1.set_title('Histogram of {title}'.format(title = title))

    plt.tight_layout()
    plt.show()

# column_name -- is the column for which histogram needs to be made 
# title-- title of the histogram 
# both column and title need to passed in string format eg - column_name is 'width'
    
hist_plot('length','length')
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_42_0.png)


The variables have been divided into 5 equal bins and the frequency in each of the bins can be seen. From the histogram above we can say that here are 70 cars with height 54 inch. Now, if we wish to see the distribution of a variable, say length, for each type of body-style, then there are two ways to do it which have described below.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Density Plot
A density plot is a **smoothed, continuous version of a histogram** estimated from the data.<br>

The x-axis is the value of the variable just like in a histogram. The y-axis in a density plot is the probability density function for the kernel density estimation. However, one should carefully specify **this is a probability density and not a probability**. The difference is the probability density is the probability per unit on the x-axis. To convert to an actual probability, we need to find the area under the curve for a specific interval on the x-axis.<br>

Because this is a probability density and not a probability, the y-axis can take values greater than one. The only requirement of the density plot is that the total area under the curve integrates to one.<br>

Think of the y-axis on a density plot as a value only for relative comparisons between different categories.

The code to make a density plot is given below:


```python
# List of body-styles to plot
body_style = auto_mobile['body-style'].unique()
fig = plt.figure(figsize = (8,8)) 
# Iterate through the five body types
for i in body_style:
    subset = auto_mobile[auto_mobile['body-style'] == i]
# Draw the density plot
    sns.distplot(subset['length'], hist = False, kde = True,
                 kde_kws = {'linewidth': 1},
                 label = i)
# Plot formatting
plt.legend(prop={'size': 10}, title = 'body-style')
plt.title('Density Plot of length with different body styles')
plt.xlabel('length')
plt.ylabel('probability density')
```




    Text(0, 0.5, 'probability density')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_46_1.png)


Hence from the plot a comparison of the distributions of length for the different styles of cars can be made. Similarly, a histogram and density plot for visualization; the code for which is given below:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


```python
#Histogram and density plot 
plt.subplots(figsize=(6,6), dpi=100)
sns.distplot( auto_mobile.loc[auto_mobile['body-style']=='hatchback', "length"] , color="gold", label="hatchback")
sns.distplot( auto_mobile.loc[auto_mobile['body-style']=='sedan', "length"] , color="yellowgreen", label="sedan")
# in a similar way, all the categories can be included in the density plot
# sns.distplot( auto_mobile.loc[auto_mobile['body-style']=='wagon', "length"] , color="lightskyblue", label="wagon")

plt.title('Length')
plt.legend();
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_49_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Boxplot
When the data distribution is displayed in a standardized way using 5 summary – minimum, Q1 (First Quartile), median, Q3(third Quartile), and maximum, it is called a Box plot.<br>

It is also termed as box and whisker plot when the lines extending from the boxes indicate variability outside the upper and lower quartiles.<br>

Outliers can be plotted as unique points.<br>


**Application of Boxplot:**
It is used to know:<br>
1. The outliers and its values<br>
2. Symmetry of Data<br>
3. Tight grouping of data<br>
4. Data skewness -if, in which direction and how<br>


```python
# function for box plot
def box_plot(column_name, title):
    l = auto_mobile[column_name]
    fig = plt.figure(figsize = (8,8))
    #adding subplots
    ax1 = fig.add_subplot(221)
    #plot

    #boxplot of the mentioned column 
    ax1.boxplot(l,notch=False, patch_artist=True,
                boxprops=dict(facecolor='gold', color='black'),
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                flierprops=dict(color='black', markeredgecolor='black'),
                medianprops=dict(color='black'))
    ax1.set_title('Boxplot of {column}'.format(column = title))

    plt.tight_layout()
    plt.show()
# function for box plot
# column_name -- is the column for which box plot need to be made 
# title-- title of the box plot 
# both column and title need to passed in string format eg - column_name is 'width'
box_plot('length', 'length')
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_52_0.png)


The statistics that can be used for univariate analysis of numerical data can be checked in a table by using the function below:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Table
Using the function  given below, we can make a table to show all the statistics that are used in bivariate analysis. Please note that using df.describe() also gives  a table of similar structure but would not include values for some statistics like skewness, variance, etc.


```python
from scipy import stats
import statistics
def univariate_numerical(data):
    df=pd.DataFrame()
    df['Count']=data.count()
    df['Minimum']=data.min()
    df['Maximum']=data.max()
    df['Mean']=data.mean()
    df['Median']=data.median()
    df['Variance']=data.var()
    df['Standard Deviation']=data.std()
    df['Skewness']=data.skew()
    df['Kurtosis']=data.kurtosis()
    return df
```


```python
df=univariate_numerical(auto_mobile)
```


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
      <th>Minimum</th>
      <th>Maximum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Variance</th>
      <th>Standard Deviation</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>symboling</th>
      <td>203</td>
      <td>-2</td>
      <td>3</td>
      <td>0.837438</td>
      <td>1.00</td>
      <td>1.562552e+00</td>
      <td>1.250021</td>
      <td>0.204275</td>
      <td>-0.691709</td>
    </tr>
    <tr>
      <th>normalized-losses</th>
      <td>203</td>
      <td>65</td>
      <td>256</td>
      <td>121.871921</td>
      <td>122.00</td>
      <td>1.010261e+03</td>
      <td>31.784599</td>
      <td>0.864408</td>
      <td>1.403077</td>
    </tr>
    <tr>
      <th>make</th>
      <td>203</td>
      <td>alfa-romero</td>
      <td>volvo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fuel-type</th>
      <td>203</td>
      <td>diesel</td>
      <td>gas</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>aspiration</th>
      <td>203</td>
      <td>std</td>
      <td>turbo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num-of-doors</th>
      <td>203</td>
      <td>four</td>
      <td>two</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>body-style</th>
      <td>203</td>
      <td>convertible</td>
      <td>wagon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <td>203</td>
      <td>4wd</td>
      <td>rwd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>engine-location</th>
      <td>203</td>
      <td>front</td>
      <td>rear</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>wheel-base</th>
      <td>203</td>
      <td>86.6</td>
      <td>120.9</td>
      <td>98.781281</td>
      <td>97.00</td>
      <td>3.649361e+01</td>
      <td>6.040994</td>
      <td>1.041170</td>
      <td>0.986065</td>
    </tr>
    <tr>
      <th>length</th>
      <td>203</td>
      <td>141.1</td>
      <td>208.1</td>
      <td>174.113300</td>
      <td>173.20</td>
      <td>1.522531e+02</td>
      <td>12.339090</td>
      <td>0.154086</td>
      <td>-0.075680</td>
    </tr>
    <tr>
      <th>width</th>
      <td>203</td>
      <td>60.3</td>
      <td>72.3</td>
      <td>65.915271</td>
      <td>65.50</td>
      <td>4.623677e+00</td>
      <td>2.150274</td>
      <td>0.900685</td>
      <td>0.687375</td>
    </tr>
    <tr>
      <th>height</th>
      <td>203</td>
      <td>47.8</td>
      <td>59.8</td>
      <td>53.731527</td>
      <td>54.10</td>
      <td>5.965932e+00</td>
      <td>2.442526</td>
      <td>0.064134</td>
      <td>-0.429298</td>
    </tr>
    <tr>
      <th>curb-weight</th>
      <td>203</td>
      <td>1488</td>
      <td>4066</td>
      <td>2557.916256</td>
      <td>2414.00</td>
      <td>2.730659e+05</td>
      <td>522.557049</td>
      <td>0.668942</td>
      <td>-0.069648</td>
    </tr>
    <tr>
      <th>engine-type</th>
      <td>203</td>
      <td>dohc</td>
      <td>rotor</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num-of-cylinders</th>
      <td>203</td>
      <td>eight</td>
      <td>two</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>engine-size</th>
      <td>203</td>
      <td>61</td>
      <td>326</td>
      <td>127.073892</td>
      <td>120.00</td>
      <td>1.746999e+03</td>
      <td>41.797123</td>
      <td>1.934993</td>
      <td>5.233661</td>
    </tr>
    <tr>
      <th>fuel-system</th>
      <td>203</td>
      <td>1bbl</td>
      <td>spfi</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bore</th>
      <td>199</td>
      <td>2.54</td>
      <td>3.94</td>
      <td>3.330955</td>
      <td>3.31</td>
      <td>7.510565e-02</td>
      <td>0.274054</td>
      <td>0.013419</td>
      <td>-0.830965</td>
    </tr>
    <tr>
      <th>stroke</th>
      <td>199</td>
      <td>2.07</td>
      <td>4.17</td>
      <td>3.254070</td>
      <td>3.29</td>
      <td>1.011384e-01</td>
      <td>0.318023</td>
      <td>-0.669515</td>
      <td>2.030592</td>
    </tr>
    <tr>
      <th>compression-ratio</th>
      <td>203</td>
      <td>7</td>
      <td>23</td>
      <td>10.093202</td>
      <td>9.00</td>
      <td>1.511822e+01</td>
      <td>3.888216</td>
      <td>2.682640</td>
      <td>5.643878</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>203</td>
      <td>48</td>
      <td>13207</td>
      <td>233.556650</td>
      <td>95.00</td>
      <td>1.684589e+06</td>
      <td>1297.917006</td>
      <td>9.985047</td>
      <td>98.770156</td>
    </tr>
    <tr>
      <th>peak-rpm</th>
      <td>201</td>
      <td>4150</td>
      <td>6600</td>
      <td>5125.870647</td>
      <td>5200.00</td>
      <td>2.302274e+05</td>
      <td>479.820136</td>
      <td>0.073094</td>
      <td>0.068155</td>
    </tr>
    <tr>
      <th>city-mpg</th>
      <td>203</td>
      <td>13</td>
      <td>49</td>
      <td>25.172414</td>
      <td>24.00</td>
      <td>4.263844e+01</td>
      <td>6.529812</td>
      <td>0.673533</td>
      <td>0.624470</td>
    </tr>
    <tr>
      <th>highway-mpg</th>
      <td>203</td>
      <td>16</td>
      <td>54</td>
      <td>30.699507</td>
      <td>30.00</td>
      <td>4.726074e+01</td>
      <td>6.874645</td>
      <td>0.549104</td>
      <td>0.479323</td>
    </tr>
    <tr>
      <th>price</th>
      <td>203</td>
      <td>5118</td>
      <td>45400</td>
      <td>13241.911330</td>
      <td>10595.00</td>
      <td>6.239354e+07</td>
      <td>7898.957924</td>
      <td>1.812335</td>
      <td>3.287412</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

# Bivariate analysis
Bivariate analysis is performed to **find the relationship between each variable in the dataset and the target variable of interest** (or) using 2 variables and finding relationship  between them. Ex:-Box plot, Violin plot.<br><br>**Bivariate Analysis** can be further classified in broad the category 
1. Numerical- Numerical
2. Categorical- Categorical 
3. Numerical- Categorical 

## Numerical-Numerical

In this the relationship between the Numerical variables is studied by plotting various plot such as scatter plot, violin plot

### Scatter plot
A scatter plot (aka scatter chart, scatter graph) uses **dots to represent values for two different numeric variables**. The position of each dot on the horizontal and vertical axis indicates values for an individual data point. Scatter plots are used to observe relationships between variables.<br><br>**When you should use a scatter plot**<br> Scatter plots’ primary uses are to observe and show relationships between two numeric variables. The dots in a scatter plot not only report the values of individual data points, but also patterns when the data are taken as a whole.



```python
# Scatter plot using Seaborn
# Findings: The more the engine size the costlier the price is
ax = sns.scatterplot(x="price", y="engine-size", data=auto_mobile)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_64_0.png)



```python
# Scatter plot using pandas
ax1 = auto_mobile.plot.scatter(x='highway-mpg',
                      y='peak-rpm',
                      c='gold', figsize = (8,8))
plt.xlabel('highway-mpg') 
plt.ylabel('Peak RPM')
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_65_0.png)



```python
fig = plt.figure(figsize = (8,8)) 
# Scatter plot using Matplotlib
plt.scatter(auto_mobile['compression-ratio'], auto_mobile['price'], alpha=0.5, c='g')
# plt.title('Scatter plot pythonspot.com')
plt.xlabel('compression-ratio')
plt.ylabel('price')
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_66_0.png)



```python
fig = plt.figure(figsize = (8,8)) 
#plot
plt.scatter(auto_mobile['engine-size'],auto_mobile['peak-rpm'])
plt.xlabel('Engine size')
plt.ylabel('Peak RPM')
```




    Text(0, 0.5, 'Peak RPM')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_67_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Categorical-Numerical

### Bar Plots
Bar plot for categorical and Numerical 


```python
fig = plt.figure(figsize = (8,4)) 
#plot
auto_mobile.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar', align='center', alpha=1, width=0.4, color='yellowgreen')
plt.title("Drive wheels City MPG")
plt.ylabel('City MPG')
plt.xlabel('Drive wheels')
```




    Text(0.5, 0, 'Drive wheels')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_71_1.png)



```python
fig = plt.figure(figsize = (8,4)) 
#plot
auto_mobile.groupby('drive-wheels')['highway-mpg'].mean().plot(kind='bar', align='center', alpha=1, width=0.4, color='lightblue');
plt.title("Drive wheels Highway MPG")
plt.ylabel('Highway MPG')
plt.xlabel('Drive wheels')
```




    Text(0.5, 0, 'Drive wheels')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_72_1.png)



```python
# Count of each number of cylenders
auto_mobile['num-of-cylinders'].value_counts()
```




    four      157
    six        24
    five       11
    eight       5
    two         4
    three       1
    twelve      1
    Name: num-of-cylinders, dtype: int64



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Box Plots
Box Plot for bivariate analysis


```python
# From here we can see the outlier with respect to our dependent variable 
# plt.rcParams['figure.figsize']=(8,8)
ax = sns.boxplot(x="num-of-cylinders", y="price", data=auto_mobile)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_76_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Categorical-Categorical

### Cross Tab
Here the relation between the categorical variables and its distribution can be viewed


```python
fig = pd.crosstab(auto_mobile['body-style'], auto_mobile['fuel-type']).plot(kind='bar', stacked=True, width = 0.4, figsize = (9,9), color=['lightblue','yellowgreen'])
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_80_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Z-Test

A Z-test is a **type of hypothesis test**. Hypothesis testing is just a way for you to figure out if results from a test are valid or repeatable.

For example, if someone said they had found a new drug that cures cancer, you would want to be sure it was probably true. A hypothesis test will tell you if it is probably true, or probably not true. A Z-test, is used when data is approximately normally distributed.

<Br>Several different types of tests are used in statistics (i.e. f test, chi square test, t test). **A Z test is used if:**
1. Sample size is greater than 30. Otherwise, a t test can be used 
2. Data points should be independent from each other. In other words, one data point isn’t related or doesn’t affect another data point
3. Data should be normally distributed. However, for large sample sizes (over 30) this does not always matter
4. Data should be randomly selected from a population, where each item has an equal chance of being selected. Sample sizes should be equal if at all possible

Interpretation depends on hypothesis, if P-Value is less than 0.05 then the null hypothesis must be rejected.

The Test Statistic: When sample is taken from a normal distribution with known variance, then our test statistic is:



```python
import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests
ztest ,pval = stests.ztest(auto_mobile['highway-mpg'])
print("if p-value is less than 0.05 than we have can reject our null hypothes is: p-valueis: {pval}".format(pval= pval))


ztest ,pval1 = stests.ztest(auto_mobile['highway-mpg'], x2=auto_mobile['price'], value=0,alternative='two-sided')
print("if p-value is less than 0.05 than we have can reject our null hypothes is: p-valueis: {pval}".format(pval= pval1))

```

    if p-value is less than 0.05 than we have can reject our null hypothes is: p-valueis: 0.0
    if p-value is less than 0.05 than we have can reject our null hypothes is: p-valueis: 1.6397793293167021e-125
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### T-Test

The t test tells how significant the differences between groups are. In other words it specifies if those differences (measured in means/averages) could have happened by chance.

T-test measures, if two samples are different from one another. One of these samples could be the population, however, T-test in place of a Z-test if the population’s standard deviation is unknown.
There are a lot of similar assumptions to the Z-test. The sample must be random and independently selected as well as drawn from the normal distribution. The values should also be numeric and continuous. The sample size does not necessarily have to be large.

<br>Interpretation depends on hypothesis, if P-Value is less than 0.05 then we must reject the null hypothesis

x̄1 is the mean of first data set<br>
x̄2 is the mean of second data set<br>
S12 is the standard deviation of first data set<br>
S22 is the standard deviation of second data set<br>
N1 is the number of elements in the first data set<br>
N2 is the number of elements in the second data set<br>


```python
# sample up wind
x1 = [10.8, 10.0, 8.2, 9.9, 11.6, 10.1, 11.3, 10.3, 10.7, 9.7, 
      7.8, 9.6, 9.7, 11.6, 10.3, 9.8, 12.3, 11.0, 10.4, 10.4]

# sample down wind
x2 = [7.8, 7.5, 9.5, 11.7, 8.1, 8.8, 8.8, 7.7, 9.7, 7.0, 
      9.0, 9.7, 11.3, 8.7, 8.8, 10.9, 10.3, 9.6, 8.4, 6.6,
      7.2, 7.6, 11.5, 6.6, 8.6, 10.5, 8.4, 8.5, 10.2, 9.2]

# equal sample size and assume equal population variance
t_critical = 1.677
N1 = len(x1)
N2 = len(x2)
d1 = N1-1
d2 = N2-1
df = d1+d2
s1 = np.std(x1,ddof=1)
s2 = np.std(x2,ddof=1)
x1_bar = np.mean(x1)
x2_bar = np.mean(x2)

sp = np.sqrt((d1*s1**2 + d2*s2**2)/df)
se = sp*np.sqrt(1/N1 + 1/N2)
t = (x2_bar - x1_bar)/(se)
print("t-statistic", t)
# a two-sample independent t-test is done with scipy as follows
# NOTE: the p-value given is two-sided so the one-sided p value would be p/2
t, p_twosided = stats.ttest_ind(x2, x1, equal_var=True)
print("t = ",t, ", p_twosided = ", p_twosided, ", p_onesided =", p_twosided/2)
print("if p-value is less than 0.05 than we have can reject our null hypothesis- p-value is: {pval}".format(pval= p_twosided))

```

    t-statistic -3.5981947686898033
    t =  -3.5981947686898033 , p_twosided =  0.0007560337478801464 , p_onesided = 0.0003780168739400732
    if p-value is less than 0.05 than we have can reject our null hypothesis- p-value is: 0.0007560337478801464
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Chi-Square Tests

The chi-square statistical test for numerical variables is used to **determine whether there is a significant difference between an expected distribution and an actual distribution**. It is typically used with categorical data such as educational attainment, colours, or gender.


```python
a1 = [6, 4, 5, 10]
a2 = [8, 5, 3, 3]
a3 = [5, 4, 8, 4]
a4 = [4, 11, 7, 13]
a5 = [5, 8, 7, 6]
a6 = [7, 3, 5, 9]
dice = np.array([a1, a2, a3, a4, a5, a6])
```


```python
from scipy import stats

stats.chi2_contingency(dice)
```




    (16.490612061288754,
     0.35021521809742745,
     15,
     array([[ 5.83333333,  5.83333333,  5.83333333,  7.5       ],
            [ 4.43333333,  4.43333333,  4.43333333,  5.7       ],
            [ 4.9       ,  4.9       ,  4.9       ,  6.3       ],
            [ 8.16666667,  8.16666667,  8.16666667, 10.5       ],
            [ 6.06666667,  6.06666667,  6.06666667,  7.8       ],
            [ 5.6       ,  5.6       ,  5.6       ,  7.2       ]]))




```python
# interpretation of results 
chi2_stat, p_val, dof, ex = stats.chi2_contingency(dice)
print("Chi2 Stat          :",chi2_stat)
print("Degrees of Freedom :",dof)
print("P-Value            :",p_val)
print('\n')
print("Contingency Table  :\n",ex)

```

    Chi2 Stat          : 16.490612061288754
    Degrees of Freedom : 15
    P-Value            : 0.35021521809742745
    
    
    Contingency Table  :
     [[ 5.83333333  5.83333333  5.83333333  7.5       ]
     [ 4.43333333  4.43333333  4.43333333  5.7       ]
     [ 4.9         4.9         4.9         6.3       ]
     [ 8.16666667  8.16666667  8.16666667 10.5       ]
     [ 6.06666667  6.06666667  6.06666667  7.8       ]
     [ 5.6         5.6         5.6         7.2       ]]
    

**Chi Square for Categorical Variables**


```python
gender = ['male','male','male','male', 'female','female','female','female','female']
Likes_shoping = ['No','Yes','Yes','Yes','Yes','Yes','No','No','No']
data = pd.DataFrame({'gender': gender, 'Likes_shoping':Likes_shoping})
```


```python
#Contingency Table
contingency_table = pd.crosstab(data['gender'], data['Likes_shoping'])
print('contingency_table :',contingency_table)
print('\n')


print("===================================================")

#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :\n",Observed_Values)

# print("==========================================================================")

#Expected Values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("\nExpected Values:\n ",Expected_Values)

# print("==========================================================================")

#Degree of Freedom
no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("\nDegree of Freedom: ",df)

# print("==========================================================================")

#Significance Level 5%
alpha=0.05
#chi-square statistic - χ2
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("\nchi-square statistic  :",chi_square_statistic)

# print("==========================================================================")

#critical_value
critical_value=chi2.ppf(q=1-alpha,df=df)
print('\ncritical_value        :',critical_value)

# print("==========================================================================")

#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('\np-value               :',p_value)
print('Significance level    :',alpha)
print('Degree of Freedom     :',df)
print('chi-square statistic  :',chi_square_statistic)
print('critical_value        :',critical_value)
print('p-value               :',p_value)

print("===================================================")
#compare chi_square_statistic with critical_value and p-value which is the probability of getting chi-square>0.09 (chi_square_statistic)
if chi_square_statistic>=critical_value:
    print("\n Critical_Value: Reject H0,There is a relationship between 2 categorical variables")
else:
    print("\n Critical_Value: Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("\n p-value       : Retain H0,There is no relationship between 2 categorical variables")
else:
    print("\n p-value       : Retain H0,There is no relationship between 2 categorical variables")
```

    contingency_table : Likes_shoping  No  Yes
    gender                
    female          3    2
    male            1    3
    
    
    ===================================================
    Observed Values :
     [[3 2]
     [1 3]]
    
    Expected Values:
      [[2.22222222 2.77777778]
     [1.77777778 2.22222222]]
    
    Degree of Freedom:  1
    
    chi-square statistic  : 1.1024999999999996
    
    critical_value        : 3.841458820694124
    
    p-value               : 0.29371811275179205
    Significance level    : 0.05
    Degree of Freedom     : 1
    chi-square statistic  : 1.1024999999999996
    critical_value        : 3.841458820694124
    p-value               : 0.29371811275179205
    ===================================================
    
     Critical_Value: Retain H0,There is no relationship between 2 categorical variables
    
     p-value       : Retain H0,There is no relationship between 2 categorical variables
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

# Multivariate Analysis

### Correlation

**Pearson’s Correlation Coefficient**<br>
Pearson’s correlation coefficient is the test statistics that measures the statistical relationship, or association, between two continuous variables.  It is known as the best method of measuring the association between variables of interest because it is **based on the method of covariance**.  It gives information about the magnitude of the association, or correlation, as well as the direction of the relationship.


```python
a = auto_mobile.corr()
plt.rcParams['figure.figsize']=(15,15)
ax = sns.heatmap(a, linewidth=0.5, cmap= 'BuGn_r', annot = True)
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_104_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Combination Chart
Its a combination of line and bar plot; and wherever time variable is involved it can be used.


```python
# combo char can be seen for nultiple vairable which depends on time example want to see the profit and sales over the period of time 
plt.rcParams['figure.figsize']=(10,10)
left_2013 = pd.DataFrame(
    {'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
               'oct', 'nov', 'dec'],
     '2013_val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 6]})

right_2014 = pd.DataFrame({'month': ['jan', 'feb'], '2014_val': [4, 5]})

right_2014_target = pd.DataFrame(
    {'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
               'oct', 'nov', 'dec'],
     '2014_target_val': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]})
fig = plt.figure(figsize = (9,9)) 
df_13_14 = pd.merge(left_2013, right_2014, how='outer')
df_13_14_target = pd.merge(df_13_14, right_2014_target, how='outer')

ax = df_13_14_target[['month', '2014_target_val']].plot(
    x='month', linestyle='-', marker='o', c='black')
df_13_14_target[['month', '2013_val', '2014_val']].plot(x='month', kind='bar',
                                                        ax=ax, color=['lightblue','yellowgreen'])


plt.show()
```


    <Figure size 648x648 with 0 Axes>



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_107_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Scatter Plot


```python
plt.rcParams['figure.figsize']=(15,15)
x = auto_mobile['city-mpg'].values
y = auto_mobile['num-of-cylinders'].values
ax4 = sns.lmplot(x ='city-mpg', y ='length',data=auto_mobile)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_110_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Pair Plot


```python
plt.rcParams['figure.figsize']=(20,10)
pp = sns.pairplot(data=auto_mobile,
                  y_vars=['price'],
                  x_vars=['bore', 'stroke','compression-ratio', 'horsepower', 'peak-rpm','city-mpg', 'highway-mpg'])
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/1.EDA/output_113_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


# Outlier Treatment
An outlier is an **observation point that is distant from other observations**. The boxplot and Scatter plot are two visualization tools that in most cases prove to be effective in outlier detection.<br>
Outlier Treatment can be done by two methods that will be explained in detail  below: <br>
1. z score 
2. IQR 

### Z Score
The __Z-score__ is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured.<br>
While calculating the Z-score we re-scale and centre the data and look for data points which are too far from zero. These data points which are way too far from zero will be treated as the outliers. In most of the cases a __threshold of 3 or -3__ is used i.e. if the Z-score value is greater than or less than 3 or -3 respectively, that data point will be identified as outliers.

There are no data points with z>3 in our data. Hence we will Download the Boston Data set to show how outlier removal can be done 


```python
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target
columns = boston.feature_names
#create the dataframe
boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
boston_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Finding the z score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(boston_df))

```


```python
#Finding the outliers
threshold = 3
# print(np.where(z > 3))
# To remove or filter the outliers and get the clean data:
boston_df_zs = boston_df[(z < 3).all(axis=1)]
boston_df_zs.shape, boston_df.shape
```




    ((415, 13), (506, 13))



Hence we see the columns containing the outliers are removed


```python
boston_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### IQR Score
The **interquartile range (IQR)**, also called the widespread  or middle 50%, or technically H-spread, is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles, or between upper and lower quartiles, **IQR = Q3 − Q1**<br>

In other words, the IQR is the first quartile subtracted from the third quartile; these quartiles can be clearly seen on a box plot on the data<br>

It is a measure of the dispersion like standard deviation or variance, but is much more robust against outliers<br>


```python
# function for removing outlier based on IQR
# it should only be used for continuous variable
#Calculating the IQR
def remove_outlier(df):
    Q1 = boston_df.quantile(0.25)
    Q3 = boston_df.quantile(0.75)
    IQR = Q3 - Q1
    #Removing outliers
    df1 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(IQR)
    return df1

remove_outlier(boston_df).head()
```

    CRIM         3.595038
    ZN          12.500000
    INDUS       12.910000
    CHAS         0.000000
    NOX          0.175000
    RM           0.738000
    AGE         49.050000
    DIS          3.088250
    RAD         20.000000
    TAX        387.000000
    PTRATIO      2.800000
    B           20.847500
    LSTAT       10.005000
    dtype: float64
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>



• This is the IQR for each column in the dataframe<br>
• The IQR scores helps in detecting the outliers outliers. 

Just like Z-score, the outliers outliers can be filtered out by keeping only valid values.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

