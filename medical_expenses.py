# import os
# os.getcwd()
# import kaggle
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files("mirichoi0218/insurance",unzip=True,path=r"./csv files")
import numpy as np
import pandas as pd
medical_df=pd.read_csv("./csv files/insurance.csv")
medical_df.info()
medical_df.describe()

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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

from sklearn.linear_model import LinearRegression
model=LinearRegression()

inputs= non_smoker_df[["age"]]
targets=non_smoker_df.charges
print("Input shape ",inputs.shape)
print("Targets shape ",targets.shape)
model.fit(X=inputs,y=targets)

#making predictions
model.predict(np.array([[23],
                        [37],
                        [61]]))
predictions=model.predict(inputs)
rmse(targets,predictions) 

##using bmi and age to predict
inputs,targets=non_smoker_df[["age","bmi"]],non_smoker_df["charges"]
model=LinearRegression().fit(inputs,targets)

predictions=model.predict(inputs)
loss=rmse(targets,predictions)
print("Loss:",loss)

##using children as well
inputs,targets=non_smoker_df[["age","bmi","children"]],non_smoker_df["charges"]
model=LinearRegression().fit(inputs,targets)

predictions=model.predict(inputs)
loss=rmse(targets,predictions)
print("Loss:",loss)


#Using the whole dataset instead of just non smokers with age bmi and children as predictors
inputs,targets=medical_df[["age","bmi","children"]],medical_df["charges"]

model=LinearRegression().fit(inputs,targets)

predictions=model.predict(inputs)
loss=rmse(targets,predictions)
print(loss)

sns.barplot(data=medical_df,x="smoker",y="charges")

smoker_codes={'no':0,'yes':1}
medical_df["smoker_code"]=medical_df.smoker.map(smoker_codes)
medical_df.charges.corr(medical_df.smoker_code)
 
 
 #looking into the sex column
sns.barplot(data=medical_df,x="sex",y="charges")

sex_codes={"female":0,"male":1}
medical_df['sex_code']=medical_df.sex.map(sex_codes)
medical_df.charges.corr(medical_df.sex_code)


inputs,targets=medical_df[["age","bmi","smoker_code","sex_code"]],medical_df["charges"]

model=LinearRegression().fit(inputs,targets)
predictons=model.predict(inputs)
loss=rmse(targets,predictions)
loss

sns.barplot(data=medical_df,x="region",y="charges")

from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()
enc.fit(medical_df[["region"]])
enc.categories_

one_hot=enc.transform(medical_df[["region"]]).toarray()
one_hot

medical_df[['northeast', 'northwest', 'southeast', 'southwest']]=one_hot
medical_df

input_cols=["age","bmi","children",'smoker_code','sex_code','northeast','northwest','southeast','southwest']
inputs,targets=medical_df[input_cols],medical_df["charges"]
model=LinearRegression().fit(inputs,targets)
predictions=model.predict(inputs)
loss=rmse(targets,predictions)
print(loss)

#Is it better to have two linear regression models for smokers and non smokers


weights_df=pd.DataFrame({
    "feature":np.append(input_cols,1),
    "weight":np.append(model.coef_,model.intercept_)
}
)
weights_df

#The weights cannot be understood correctly for numerical columns because some columns may have more number of possible values so they can have a larger impact even though their weights are small

#To clearly understand the meaning of weight we have to standarize the data

#THis willnotmake the model more accurate

#z=(x-mew)/sigma

from sklearn.preprocessing import StandardScaler
numeric_cols=["age","bmi",'children']
scaler=StandardScaler()
scaler.fit(medical_df[numeric_cols])
scaler.mean_
scaler.var_

scaled_inputs=scaler.transform(medical_df[numeric_cols])
scaled_inputs

cat_cols=['smoker_code','sex_code','northeast','northwest','southeast','southwest']
categorical_data=medical_df[cat_cols].values

inputs=np.concatenate((scaled_inputs,categorical_data),axis=1)
targets=medical_df.charges
model=LinearRegression().fit(inputs,targets)

predictions=model.predict(inputs)
loss=rmse(targets,predictions)
loss


weights_df=pd.DataFrame({
    "feature":np.append(input_cols,1),
    "weight":np.append(model.coef_,model.intercept_)
}
)
weights_df

