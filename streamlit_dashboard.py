import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('CleanDF.csv')

#isAdult, averageRating, runtimeMinutes, (Year?), genres (get_dummies)
# columns_of_interest = ['isAdult', 'runtimeMinutes', 'averageRating']
# X= df[columns_of_interest]
# distanceKNN = NearestNeighbors(n_neighbors = 5).fit(X)
# title = df['originalTitle'][6]
# coord = distanceKNN.kneighbors(df.loc[df['originalTitle']==title, columns_of_interest])

# for i in range(0, 5):
#     print('TOP', i+1, ':' , df['originalTitle'].iloc[coord[1][0][i]])
    

title = st.selectbox('What is your favourite title ?',
    (list(df['originalTitle'])))