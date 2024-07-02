#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pclass = st.sidebar.selectbox('Passenger class',('1','2','3'))
    SibSp = st.sidebar.selectbox('No. of siblings/Spouses',('0','1','2','3','4'))
    Parch = st.sidebar.selectbox(' Number of Parents/Children ',('0','1','2','3'))
    Age = st.sidebar.number_input("Insert the Age")
    Fare = st.sidebar.number_input("Insert the Fare")
    data = {'Pclass':Pclass,
            'SibSp':SibSp,
            'Parch':Parch,
            'Age':Age,
            'Fare':Fare}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

train_data = pd.read_csv(r"C:\Users\nitin\Downloads\Latest DS material\Deploy-Streamlit\Titanic_train.csv")
train_data.drop(["PassengerId"],inplace=True,axis = 1)
train_data.drop(["Name"],inplace=True,axis = 1)
train_data.drop(["Sex"],inplace=True,axis = 1)
train_data.drop(["Ticket"],inplace=True,axis = 1)
train_data.drop(["Cabin"],inplace=True,axis = 1)
train_data.drop(["Embarked"],inplace=True,axis = 1)
train_data = train_data.dropna()

X = train_data.values[:,1:]
Y = train_data.values[:,0]

classifier = LogisticRegression()
classifier.fit(X,Y)

prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.subheader('Predicted Result')
st.write('Passenger will survive' if prediction_proba[0][1] > 0.5 else 'Passenger will NOT survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)

