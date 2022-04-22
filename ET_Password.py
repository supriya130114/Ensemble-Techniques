# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:09:22 2022

@author: ankush
"""

import pandas as pd
import matplotlib.pyplot as plt
# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
# For creating a pipeline
from sklearn.pipeline import Pipeline
# Classifier Model (Decision Tree)
from sklearn.tree import DecisionTreeClassifier
# Read the File
data = pd.read_excel('C:\\Users\\ankush\\Desktop\\DataSets\\ET\\Ensemble_Password_Strength.xlsx')
data.describe()
data.head()
# Features which are passwords
# Selecting all rows and coloumn 1 which are passwords of type 'string'.

features = data.values[:, 0].astype('str')
print(features)
# Labels which are strength of password
# Selecting all rows and last coloumn which are passwords strengths of type 'int'.

labels = data.values[:, 1].astype('int')
print(labels)
# Sequentially apply a list of transforms and a final estimator
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('decisionTree',DecisionTreeClassifier()),
])

# Fit the Model
classifier_model.fit(features, labels)
# Instead of splitting dataset into training and testing, we keep test dataset as seprate .csv file 
df= pd.read_csv('C:\\Users\\ankush\\Desktop\\cleanpasswordlist.csv')

X = df.values[:,0].astype('str')
y = df.values[:, 1].astype('int')
print('Testing Accuracy: ',classifier_model.score(X, y)*100)
#showing predication for 50 passwords as a sample

list=features[40:90]
predict=classifier_model.predict(list)
predict
print(list)
# Taking sample of 50 passwords for ploting on Graph

x=features[100:150]
y=classifier_model.predict(x)

# Ploting graph

plt.scatter(x, y, color = 'red')
plt.title('Password vs Strength')
plt.xlabel('Password String')
plt.ylabel('Strength scale')
plt.show()
# Printing x coordinate

print(x)
# Printing y coordinate

print(y)
