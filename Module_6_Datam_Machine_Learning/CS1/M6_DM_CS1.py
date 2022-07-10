# Author: Gaurav
# Domain – Movies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.options.display.float_format = '{:.2f}'.format
data = pd.read_csv("prisoners.csv", low_memory=False)  # not a good practice
df = pd.DataFrame(data)

#############################################################
# Problem 1 (a):Load the dataset “prisoners.csv” using pandas and display the first and last five rows in the dataset.
print(df.head())
print(df.tail())

# Problem 1 (b): Use describe method in pandas and find out the number of columns.
df.describe()
print(df.columns)


#############################################################
# Problem 2 (a):Create a new column -’total_benefitted’ that is a sum of inmates benefitted through all modes.
df['total_benefitted'] = df['No. of Inmates benefitted by Elementary Education'] + df['No. of Inmates benefitted by Adult Education'] + df['No. of Inmates benefitted by Higher Education'] + df['No. of Inmates benefitted by Computer Course']
print(df.head())

#############################################################
# Problem 2 (b):Create a new row -“totals” that is the sum of all inmates benefitted through each mode across all states.

dic = dict(df.sum(axis=0, skipna = True))
dic["STATE/UT"] = 'Total'
dic["YEAR"] = 0
df2 = pd.DataFrame(dic, index={len(df)+1})
print(df2)
df = pd.concat([df,df2])
print(df)


#############################################################
# Problem 3 (a):Make a bar plot with each state name on the x -axis and their total benefitted inmates astheir bar heights. Which state has the maximum number of beneficiaries?

df = df[:-1]
print(df)
plt.bar(df["STATE/UT"],df["total_benefitted"])
# plt.show()


#############################################################
# Problem 3 (b): Make a pie chart that depicts the ratio among different modes of benefits.

# tick_label = df2.columns
tick_label = ['No. of Inmates benefitted by Elementary Education', 'No. of Inmates benefitted by Adult Education', 'No. of Inmates benefitted by Higher Education','No. of Inmates benefitted by Computer Course']
# print(tick_label)
df3 = df2
df3.pop("STATE/UT")
df3.pop("YEAR")
df3.pop("total_benefitted")
# "YEAR",'total_benefitted']
print(df3)
df3 = df3.transpose()
pd.DataFrame.to_csv(df3, '../CS2/TEst.csv')
df3 = pd.read_csv("../CS2/TEst.csv", low_memory=False, skiprows= 1)  # not a good practice
df3.columns = ['Benefit_type','Total']
print(df3)
tick_label = df3["Benefit_type"]
plt.pie(df3["Total"], labels = tick_label )
plt.show()
