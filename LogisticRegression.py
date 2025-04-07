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

train_inputs[numeric_cols]=scaler.transform(train_inputs[numeric_cols])


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



#Saving the processed data to the disk so that whenever needed we dont have to repeat the preprocessing steps all again
#Parquet format is a fast and efficient format for saving and loading pandas dataframe
 
#Creating a directory for the data 

import os
# os.getcwd()
# os.mkdir("LogisticRegressionData")


train_inputs.to_parquet('LogisticRegressionData/train_inputs.parquet')

val_inputs.to_parquet('LogisticRegressionData/val_inputs.parquet')

test_inputs.to_parquet('LogisticRegressionData/test_inputs.parquet')

pd.DataFrame(train_targets).to_parquet("LogisticRegressionData/train_targets.parquet")


pd.DataFrame(val_targets).to_parquet("LogisticRegressionData/val_targets.parquet")


pd.DataFrame(test_targets).to_parquet("LogisticRegressionData/test_targets.parquet")


#Reading from the parquet

train_inputs=pd.read_parquet("LogisticRegressionData/train_inputs.parquet")

val_inputs=pd.read_parquet("LogisticRegressionData/val_inputs.parquet")

test_inputs=pd.read_parquet("LogisticRegressionData/test_inputs.parquet")

train_targets=pd.read_parquet("LogisticRegressionData/train_targets.parquet")

val_targets=pd.read_parquet("LogisticRegressionData/val_targets.parquet")

test_targets=pd.read_parquet("LogisticRegressionData/test_targets.parquet")

train_inputs.isna().sum()
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(train_inputs[numeric_cols+encoded_cols],train_targets)

print(numeric_cols+encoded_cols)
print(model.coef_.tolist())

weigth_df=pd.DataFrame({
   "feature":(numeric_cols+encoded_cols+["intercept"]),
   "weight":np.append(model.coef_.tolist()[0],model.intercept_)
}
)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,50))


#Plotting the top 10 values with highest weights

sns.barplot(data=weigth_df.sort_values("weight",ascending=False).head(10),x="weight",y="feature",hue="feature",palette="muted",width=1)

X_train=train_inputs[numeric_cols+encoded_cols]
X_val=val_inputs[numeric_cols+encoded_cols]
X_test=test_inputs[numeric_cols+encoded_cols]

train_preds=model.predict(X_train)
train_preds

#We can output probabilistic predictions using predict_proba

train_probs=model.predict_proba(X_train)
train_probs

from sklearn.metrics import accuracy_score
accuracy_score(train_targets,train_preds)

from sklearn.metrics import confusion_matrix
confusion_matrix(train_targets,train_preds,normalize="true")

#Has a lot of false negatives therefore there is a high chance that when model predicts it is not going to rain it might rain

#Could be due to the fact that the original data has a lot of "Yes" values

def predict_and_plot(inputs,targets,name=''):
   preds=model.predict(inputs)
   accuracy =accuracy_score(targets,preds)
   print(f"Accuracy of the model is {accuracy*100}")
   cf=confusion_matrix(targets,preds,normalize='true')
   plt.figure()
   sns.heatmap(cf,annot=True)
   plt.xlabel("Prediction")
   plt.ylabel("Target")
   plt.title(f"{name} Confusion matrix")
   return preds


train_preds=predict_and_plot(X_train,train_targets,"Training")

val_preds=predict_and_plot(X_val,val_targets,"Validation")

test_preds=predict_and_plot(X_test,test_targets,"Test")

#Comparing the model against dumb models

def random_guess(inputs):
   return np.random.choice(["No","Yes"],len(inputs))

def all_no(inputs):
   return np.full(len(inputs),"No")

accuracy_score(test_targets,random_guess(X_test))
accuracy_score(test_targets,all_no(X_test))

#Our model is better than both of these

new_input=train_df.iloc[np.random.randint(0,len(train_df))]
new_input_df=pd.DataFrame([new_input])


#Transforming the new input according to how the input data for model training was transformed
new_input_df[numeric_cols]=imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols]=scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols]=encoder.transform(new_input_df[categorical_cols])

X_new_input=new_input_df[numeric_cols+encoded_cols]
X_new_input


#making a prediction
prediction = model.predict(X_new_input)
prediction

prob=model.predict_proba(X_new_input)
prob

#Creating a helper function to predict new inputs

def predict_input(single_input):
   input_df=pd.DataFrame([single_input])
   input_df[numeric_cols]=imputer.transform(input_df[numeric_cols])
   input_df[numeric_cols]=scaler.transform(input_df[numeric_cols])
   input_df[encoded_cols]=encoder.transform(input_df[categorical_cols])
   X_input=input_df[numeric_cols+encoded_cols]
   pred=model.predict(X_input)[0]
   prob=model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
   return pred,prob


new_input=train_df.iloc[np.random.randint(0,len(train_df))]
new_input
prediction,prob=predict_input(new_input)
prediction,prob

#Saving and loading trained models
import joblib

#Creating a dictionary containing all the rquired objects
aus_rain={
   'model':model,
   'imputer':imputer,
   'scaler':scaler,
   'encoder':encoder,
   'input_cols':input_cols,
   'target_cols':numeric_cols,
   'categorical_cols':categorical_cols,
   'encoded_cols':encoded_cols
}
os.getcwd()
joblib.dump(aus_rain,"aus_rain.joblib")

aus_rain2=joblib.load("aus_rain.joblib")
aus_rain2