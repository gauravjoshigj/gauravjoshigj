# Author: Gaurav
# Domain – Cereal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:.2f}'.format
data = pd.read_csv("cereal.csv", low_memory=False)  # not a good practice
df = pd.DataFrame(data)

#############################################################
# Problem 1 :Load the data from “cereal.csv” and plot histograms of sugar and vitamin content across different cereals.

hst = df[['sugars']]
plt.hist(hst, bins = 10)
# plt.show()

hst = df[['vitamins']]
plt.hist(hst, bins = 10)
# plt.show()


#############################################################
# Problem 2 :The names of the manufactures are coded using alphabets, create a new column with their fullname using the below mapping.

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

d = {'N': 'Nabisco','Q': 'Quaker Oats','K': 'Kelloggs','R': 'Raslston Purina','G': 'General Mills' ,'P' :'Post' ,'A':'American Home Foods Products'}
print(d['N'])

df['Mfg'] = df['mfr'].apply(lambda x : d[x])
# print(df)
df2 = pd.DataFrame(df[["Mfg","name"]].groupby("Mfg", as_index= False).count())
# print(df2)
df2.columns = ['Mfg','cereal_count']
# pd.DataFrame.to_csv(df2, 'TEst.csv')
plt.bar(df2["Mfg"],df2["cereal_count"])
plt.show()

#############################################################
# Problem 3 :Linear regression
print(df.dtypes)
data = pd.read_csv("cereal.csv", low_memory=False)  # not a good practice
df = pd.DataFrame(data)
print(df.shape)

x = df.iloc[:, 3:15]
y = df['rating']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)

lm= LinearRegression()
model = lm.fit(x_train,y_train)
pred_y = lm.predict(x_test)
plt.scatter(y_test,pred_y)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()