### Polynomial Features Project

For this project we will create an algorithm that take in a pandas dataframe and creates polynomial (higher degree)  features from a numeric column, and returns a new dataframe with the polynomial features.

#### Math Background

Given a numerical feature, we may want to create higher degree features. So given a value $x$, we want an algorithm that returns the tuple $( x, x^2, x^3, ..., x^p)$.

Now, given a dataframe $df1$, and a polynomial degree $p$, and a numerical column which we wish to transform, we want an algorithm that returns a dataframe $df2$ with all of the original features from $df1$, and concatenated on, the new higher degree features, up to and including degree p. This algorithm will also keep the column to be transformed and its transformed columns all adjacent to eachother in the resulting dataframe $df2$. 

### Code


```python
import numpy as np
import pandas as pd
```


```python
# So we want construct a function that takes in a dataframe
# and returns a new one with the specified poly features.

# (df, column_name, highest number of degress we want) |---> new df with poly features




def create_poly_from_df(df,column_name,p):
    
    
        
    
    
    df_copy = df.copy()
    
    # so the first thing we need to do is take in the dataframe 
    # and isolate the chosen feature.
    
    data_col = df[column_name].values.copy()
    
    #Now we want to run a loop that creates each polynomial feature.
    
    features = []
    
    new_feature_names = []
    
    
    if p == 1:
        return df
    
    
    
    for i in range(1,p+1):
        
        features.append(data_col**i)   #creating the polynomial features.
        
        
        
        # Creating the names of our new columns.
        if i == 1:
            new_feature_names.append(column_name)
        else: 
            new_feature_names.append(column_name + f'^{i}')
        
        
        
    poly_df = pd.DataFrame(np.array(features).T)   #Our dataframe of polynomial features.
    
  
    poly_df.columns = new_feature_names #Assigning column names
    poly_df.index = df_copy.index #Assigning the index of the inputted dataframe.
    
    
    if type(df) == pd.core.series.Series:
        return  poly_df #Dropping the feature from new_df because it is
    # included in the new dataframe.
     
        
        
        
        
        
        
    df_copy.drop(column_name,inplace = True,axis = 1) 
    
    
    return pd.concat([df_copy,poly_df],axis = 1) #Concatonating or old dataframe to the new one.




    
    
        
    
    
    
    
```

# Example

We can apply our function to the iris datasets sepal_length feature.


```python
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

```


```python
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
create_poly_from_df(iris,'sepal_length',3).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
      <th>sepal_length</th>
      <th>sepal_length^2</th>
      <th>sepal_length^3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>5.1</td>
      <td>26.01</td>
      <td>132.651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>4.9</td>
      <td>24.01</td>
      <td>117.649</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>4.7</td>
      <td>22.09</td>
      <td>103.823</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>4.6</td>
      <td>21.16</td>
      <td>97.336</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>5.0</td>
      <td>25.00</td>
      <td>125.000</td>
    </tr>
  </tbody>
</table>
</div>



We see our new dataframe has sepal_length, sepal_length squared, and sepal_length cubed.

Now we see how we can create high degree features.


```python
create_poly_from_df(iris[['sepal_length']],'sepal_length',6).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_length^2</th>
      <th>sepal_length^3</th>
      <th>sepal_length^4</th>
      <th>sepal_length^5</th>
      <th>sepal_length^6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>26.01</td>
      <td>132.651</td>
      <td>676.5201</td>
      <td>3450.25251</td>
      <td>17596.287801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>24.01</td>
      <td>117.649</td>
      <td>576.4801</td>
      <td>2824.75249</td>
      <td>13841.287201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>22.09</td>
      <td>103.823</td>
      <td>487.9681</td>
      <td>2293.45007</td>
      <td>10779.215329</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>21.16</td>
      <td>97.336</td>
      <td>447.7456</td>
      <td>2059.62976</td>
      <td>9474.296896</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>25.00</td>
      <td>125.000</td>
      <td>625.0000</td>
      <td>3125.00000</td>
      <td>15625.000000</td>
    </tr>
  </tbody>
</table>
</div>


