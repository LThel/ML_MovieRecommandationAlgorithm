import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

import regex as re
import random

moviedf = pd.read_csv("rotten_tomatoes_movies.csv")
criticsdf = pd.read_csv("rotten_tomatoes_critic_reviews.csv")

df = pd.read_csv('test_over500.csv')

score_weight = 3
votes_weight = 0.01

def clean_title(a):
    return re.sub("[^a-zA-Z0-9 ]","",a)

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = moviedf.iloc[indices].iloc[::-1]
    
    return results

equivalents = {
    "A+": 5,
    "A": 4.7058823528,
    "A-": 4.41176470575,
    "B+": 4.1176470587,
    "B": 3.82352941165,
    "B-": 3.5294117646,
    "C+": 3.23529411755,
    "C": 2.9411764705,
    "C-": 2.64705882345,
    "D+": 2.3529411764,
    "D": 2.05882352935,
    "D-": 1.7647058823,
    "E+": 1.47058823525,
    "E": 1.1764705882,
    "E-": 0.88235294115,
    "F+": 0.5882352941,
    "F": 0
    }

def standardize(rev):  # some of them are /4 and some /5. Better eval?
    if "/" in str(rev):
        return float(rev[:rev.index("/")])
    else:
        score = equivalents.get(rev)
        if score is not None:
            return score
        else:
            return None
        
def standardize2(rev):
    if rev > 5:
        return rev / 20

def dubber(link):
    filt = moviedf[moviedf['rotten_tomatoes_link'] == link]
    # Get the Rotten Tomatoes link from the filtered row 
    a= filt['movie_title'].values[0]
    return a

#Little bit of cleaning
criticsdf.dropna(subset="review_score",inplace=True)
criticsdf["review_score"] = criticsdf["review_score"].apply(standardize)
criticsdf.drop("review_content",axis=1,inplace=True)
criticsdf.drop("review_date",axis=1,inplace=True)
criticsdf.reset_index(inplace=True)

columns_titles = ["critic_name","top_critic","publisher_name","rotten_tomatoes_link","review_score","review_type"]
criticsdf=criticsdf.reindex(columns=columns_titles)
criticsdf.dropna(axis=0,inplace=True)
criticsdf['review_score'] = criticsdf['review_score'].round().astype(int)


df = pd.read_csv('test_over500.csv')  # Is it me or there are multiple Horror columns in this dataframe? 



#Presentation of the dataframe
dashboard = st.sidebar.radio(
    "What dashboard do you want to see ?",

    ('Data visualisation', 'Our recommandation algorithm',"Hate"))


if dashboard == 'Data visualisation':
     st.write('Not done yet')
elif dashboard == 'Our recommandation algorithm':
    st.title('Welcome to our recommandation algorithm page !')
    # Recommandation algorithm page     
    movie_title = st.selectbox("Please enter your favourite movie's title : ", (list(df['originalTitle'])))

    num_sim = st.slider('How many similar movies do you want to see?', 1, 10, 5)

    columns_of_interest = ['isAdult', 'runtimeMinutes', 'averageRating'] + list(df.iloc[:, -26:].columns)  
    X= df[columns_of_interest]
    distanceKNN = NearestNeighbors(n_neighbors = num_sim+1).fit(X)
    coord = distanceKNN.kneighbors(df.loc[df['originalTitle']==movie_title, columns_of_interest])
    for i in range(1, num_sim+1):
        st.write('TOP', i, ':' , df['originalTitle'].iloc[coord[1][0][i]],'(',str(df['startYear'].iloc[coord[1][0][i]]),')')
elif dashboard == 'Hate':
    st.title("Pick a movie you hate!")
    movie_title = st.selectbox("Please enter your least favourite movie's title : ", (list(moviedf['movie_title'])))
    filtered_df = moviedf[moviedf['movie_title'] == movie_title]
    # Get the Rotten Tomatoes link from the filtered row 
    mov= filtered_df['rotten_tomatoes_link'].values[0]

    moviehaters = criticsdf[(criticsdf['rotten_tomatoes_link'] == mov) &
    ((criticsdf['review_score'] == 1) | (criticsdf['review_score'] == 2))]
    #movieboys = criticdummies[(criticdummies['review_type'] == "Fresh")]
    crits = criticsdf["critic_name"].values.tolist()
    chosenones = criticsdf.loc[(criticsdf['critic_name'].isin(crits))]
    chosenones["review_score"] = chosenones["review_score"].apply(standardize2)
    average_scores = chosenones.groupby('rotten_tomatoes_link')['review_score'].mean()
    num_votes = chosenones.groupby('rotten_tomatoes_link')['review_type'].count()
    chosenflicks = pd.DataFrame({'rotten_tomatoes_link': chosenones['rotten_tomatoes_link'].unique()})
    chosenflicks['AverageScore'] = chosenflicks['rotten_tomatoes_link'].map(average_scores)
    chosenflicks['NumVotes'] = chosenflicks['rotten_tomatoes_link'].map(num_votes)
    # How about you just do the sum of the votes? avg * numvotes, avg, number of votes
    chosenflicks['ObjectiveScore'] = chosenflicks.apply(lambda row: (score_weight * row['AverageScore']) + (votes_weight * row['NumVotes']) + random.uniform(-0.6, 0.6), axis=1)
    #chosenflicks['rotten_tomatoes_link'] = chosenflicks['rotten_tomatoes_link'].apply(dubber)
    chosenflicks = chosenflicks.sort_values(['ObjectiveScore'], ascending=False)
    st.dataframe(data=chosenflicks, width=None, height=None, use_container_width=True)
