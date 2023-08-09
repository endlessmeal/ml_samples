import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

# set state of reproducibility for random values
RANDOM_STATE = 42

# load dataset
Data = pd.read_csv(
    "https://github.com/datasciencedojo/datasets/blob/master/titanic.csv",
    sep=";"
    )

# 891 objects and 12 features
print(Data.shape) # (891, 12)

# get numeral features with its std, mean, count etc.
print(Data.describe())

# get only features with values that can be divided by categories
print(Data.describe(include='object'))

# count all objects by categories (Male, Female, Unknown) 
print(Data['Sex'].value_counts())

# remove all unkown sex
Data = Data[Data['Sex'] != 'unknown']

# encode features to 1 and 0 by male and female states
Data['Sex'] = Data['Sex'].map({'male': 1, 'female': 0})

# check how sex affect on survival 
sns.barplot(x='Sex', y='Survived', data=Data, palette='summer')
plt.title('Sex - Survived')
plt.show()

# pclass by survived correlation
sns.barplot(x='Pclass', y='Survived', data=Data, palette='summer')
plt.title('Pclass - Survived')
plt.show()

# pclass and sex by survived
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=Data, palette='autumn')
plt.title('Sex - Survived')
plt.show()

# age feature has many null values so it need to be completed by its median values
median_age = Data['Age'].median()
Data['Age'].fillna(median_age, inplace=True)

# age distribution of passengers
sns.displot(Data['Age'], kde=True)
plt.show()

# correlation matrix
corr = Data.corr()
sns.heatmap(corr, annot=True)

# one by one dependencies of features
g = sns.pairplot(
    Data,
    hue='Survived',
    palette = 'seismic',
    height=4,
    diag_kind = 'kde',
    diag_kws=dict(fill=True),
    plot_kws=dict(s=50)
    )

g.set(xticklabels=[])

# ADDITIONAL

# create three new features
Data['NameLen'] = [len(i) for i in Data['Name']]
Data['FamilySize'] = [x + y + 1 for x, y in zip(Data['Parch'], Data['SibSp'])]
Data['IsAlone'] = [1 if x > 1 else 0 for x in Data['FamilySize']]

# correlation matrix
Data.drop(columns='PassengerId', inplace=True)
corr = Data.corr()
sns.heatmap(data=corr, annot=True)

# fuck