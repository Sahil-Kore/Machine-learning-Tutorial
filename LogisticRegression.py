# import kaggle 
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files("jsphyg/weather-dataset-rattle-package",path="./csv files",unzip=True)


import pandas as pd
raw_df=pd.read_csv("csv files/weatherAUS.csv")

raw_df
raw_df.info()

raw_df.dropna(subset=["RainToday","RainTomorrow"],inplace=True)
raw_df.info()

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=4
matplotlib.rcParams['figure.figsize']=(10,6)
matplotlib.rcParams['figure.facecolor']='#00000000'

plt.title("Location vs Rainy Days")
sns.histplot(raw_df,x="Location",hue="RainToday")

#Rain depends on region
px.histogram(raw_df,x="Location",title="Location vs Rainy Days",color="RainToday")

#Rain depends on temperature
px.histogram(raw_df,x="Temp3pm",title="Temperature vs Rainy Days",color="RainTomorrow")



#Rain today and Raintomorrow
px.histogram(raw_df,x="RainToday",title="RainTomorrow vs RainToday",color="RainTomorrow")

sns.histplot(raw_df,x="RainToday",hue="RainTomorrow",multiple="stack")

# sns.scatterplot(raw_df.sample(2000),x="MinTemp",y='MaxTemp',hue="RainToday")

#When the temperature does not have much variation then it raintoday is today



#When temperature is low and humidity is high it rains

#sns.scatterplot(raw_df.sample(2000),x="Temp3pm",y='Humidity3pm',hue="RainTomorrow")


# sns.pairplot(raw_df.drop("Date",axis=1).sample(2000),hue="RainTomorrow") 




#When working with massive datasets containing million of rows it's a good idea to work with a sample intially to quickly set up your model training notebook.If you'd like to work with a sample just set the value of use_sample =True

use_sample=False
sample_fraction=0.1

if use_sample:
   raw_df=raw_df.sample(frac=sample_fraction).copy()  
   
from sklearn.model_selection import train_test_split
train_val_df,test_=train_test_split(raw_df,test_size=0.2,random_state=42)
train_df,val_df=train_test_split(train_val_df,test_size=0.25,random_state=42)


##however while working with dates its often better idea to separeate the training ,validation ,and test sets with time so that the model is trained o ndata from the past and evaluated on data from the future

sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year,palette="viridis")
year=pd.to_datetime(raw_df.Date).dt.year
train_df=raw_df[year<2015]
val_df=raw_df[year==2015]
test_df=raw_df[year>2015]

print("train_df.shape:",train_df.shape)
print("val_df.shape:",val_df.shape)

#removing the first(date) and last(target) column
input_cols=list(train_df.columns)[1:-1]
input_cols

target_col="RainTomorrow"

train_inputs=train_df[input_cols].copy()
train_targets=train_df[target_col].copy()

val_inputs=val_df[input_cols].copy()
val_targets=val_df[target_col].copy()

test_inputs=test_df[input_cols].copy()
test_targets=test_df[target_col].copy()

import numpy as np

#separating numeric and categorical columns
numeric_cols=train_inputs.select_dtypes(include=np.number).columns.to_list()

categorical_cols=train_inputs.select_dtypes('object').columns.to_list()

train_inputs[numeric_cols].describe()

train_inputs[categorical_cols].nunique()



#Imputation:-The process of filling missing values

#There are several techniques for impputation but we'll use the most basic one replacing missing values with the average value inthe column using the SimpleImputer class from sklearn.impute


from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy="mean")


#Before imputing lets check the no. of missing values in each numeric column

train_inputs[numeric_cols].isna().sum()

imputer.fit(raw_df[numeric_cols])

imputer.statistics_

train_inputs[numeric_cols]=imputer.transform(train_inputs[numeric_cols])

val_inputs[numeric_cols]=imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]=imputer.transform(test_inputs[numeric_cols])

train_inputs[numeric_cols].isna().sum()



#Another good practice is to scale the numeric input features to small range of eg (0,1) or (-1,1).Scaling numeric features ensures that no particular feature has a disproportionate impact on the models loss.Optimization algorithms also work better in pratice with smaller numbers

#using minmax scaler form sklearn to scale value to the range of (0,1)

from sklearn.preprocessing import MinMaxScaler
MinMaxScaler?

scaler=MinMaxScaler()
scaler.fit(raw_df[numeric_cols])
scaler.data_min_

train_inputs[numeric_cols]=scaler.transform(train_df[numeric_cols])

val_inputs[numeric_cols]=scaler.transform(val_inputs[numeric_cols])

test_inputs[numeric_cols]=scaler.transform(test_inputs[numeric_cols])


#Encoding Categorical data 
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')


encoder.fit(raw_df[categorical_cols])
encoder.categories_
encoded_cols=list(encoder.get_feature_names_out())
encoded_cols


train_inputs[encoded_cols]=encoder.transform(train_inputs[categorical_cols])

val_inputs[encoded_cols]=encoder.transform(val_inputs[categorical_cols])

test_inputs[encoded_cols]=encoder.transform(test_inputs[categorical_cols])

print(train_inputs.columns)
