import kaggle 
kaggle.api.authenticate()
kaggle.api.dataset_download_files("jsphyg/weather-dataset-rattle-package",path="./csv files",unzip=True)


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

sns.scatterplot(raw_df.sample(2000),x="MinTemp",y='MaxTemp',hue="RainToday")

#When the temperature does not have much variation then it raintoday is today



#When temperature is low and humidity is high it rains
sns.scatterplot(raw_df.sample(2000),x="Temp3pm",y='Humidity3pm',hue="RainTomorrow")


sns.pairplot(raw_df.drop("Date",axis=1).sample(2000),hue="RainTomorrow") 

