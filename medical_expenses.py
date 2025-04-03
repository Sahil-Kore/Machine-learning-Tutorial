# import os
# os.getcwd()
# import kaggle
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files("mirichoi0218/insurance",unzip=True,path=r"./csv files")


import pandas as pd
medical_df=pd.read_csv("./csv files/insurance.csv")
medical_df.info()
medical_df.describe()

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,6)
matplotlib.rcParams['figure.facecolor']="#00000000"


#Creating a histogram to plot the age column as a histogram

fig=px.histogram(medical_df,x="age",marginal="box",#Plot a box plot above the histogram
                 nbins=47,title="Distribution of age")
fig.update_layout(bargap=0.1)
fig.show()

sns.histplot(data=medical_df,x="age",bins=47)

#Why there are over twice the number of people in the age 18
#Might be because there may be discount for ages around 18
#18 is the legal age so many apply as the get they turn 18


#Distribution of bmi
fig=px.histogram(medical_df,x="bmi",marginal="box",#Plot a box plot above the histogram
                 nbins=47,title="Distribution of age")
fig.update_layout(bargap=0.1)
fig.show()

sns.histplot(data=medical_df,x="bmi",bins=47)

#Looks like a normal distribution


#Distribution on charges
#splitting the data if the person based on the fact if the person is smoker or not
sns.histplot(data=medical_df,x="charges",hue="smoker")

#For most of the people the annual medical charges are below 10000
#Non smokers are usually having lower charges
#For most customers the annual medical charges are under $10000 only a small fraction of customers have higer medical expenses possibly due to accidents major illnesses and genetic diseases .The distribution folows power law

sns.histplot(medical_df,x="smoker",hue="sex")

sns.scatterplot(medical_df,x="age",y="charges",hue="smoker") 

#For every age it can be seen the smokers have higher charges
#Two section can be seen could be smokers and heavy smokers


sns.scatterplot(medical_df,x="bmi",y="charges",hue="smoker") 

#For non smokers there is no definite realtion between bmi and charges 
#But for smokers people with high bmi have high charges

sns.scatterplot(medical_df,x="children",y="charges")
#Better to use violin plot for discrete values
sns.violinplot(medical_df,x="children",y="charges")

#As the bulk of the violin plot is going us as the number of children increase


#Correlation coefficient 
medical_df.charges.corr(medical_df.age)
medical_df.charges.corr(medical_df.bmi)
medical_df.charges.corr(medical_df.children)

#To compute correlation for categorical data they must be converted to numeric columns
smoker_values={'no':0,"yes":1}
smoker_numeric=medical_df.smoker.map(smoker_values)
smoker_numeric

medical_df.charges.corr(smoker_numeric)

#plotting a correalation matrix
sns.heatmap(medical_df.corr(numeric_only=True),cmap="Blues",annot=True)

non_smoker_df=medical_df[medical_df.smoker=="no"]

sns.scatterplot(data=non_smoker_df,x="age",y="charges")