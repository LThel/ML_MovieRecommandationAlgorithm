#================================= Movie recommandation system ================================================
#====================== The hate-crush algorithm - Streamlit dashboard ========================================
# Here you can find the code of our Streamlit. On the Streamlit you have a brief presentation of the datas and 2 dashboards to test our 2 algorithms.
# Technologies used : Python (Sk-learn, Pandas, Numpy, Matplotlib&Seaborn), Streamlit
# Team :
# Antonio - Junior Data Analyst
# Louis - Junior Data Analyst - Open to work or to join other collaborative projects !(Linkedin : www.linkedin.com/in/louisthellier; CV : https://drive.google.com/file/d/1vJm6Jv-W9RXXKj9dQcOAraonT1LXfUqI/view?usp=sharing); Github : https://github.com/LThel;)


#Import the libraries
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import regex as re
import random
import bar_chart_race as bcr

#Import the datas
# df = pd.read_csv('test_over500.csv')
# df = pd.read_csv('test_over500_USnonUS.csv')
# df = pd.read_csv('test_over500_over5translation.csv')
df = pd.read_csv("dataframes\maxi_filtered_df3000.csv")
df_US = df[df["origin"] == "US"]
df_NoUS = df[df["origin"] == "Not US"]
df_titles = pd.read_csv("dataframes\titles.csv")
df_ov500 = pd.read_csv("dataframes\test_over500.csv")
df_ov500 = df_ov500.iloc[:, 0:42]
moviedf = pd.read_csv("dataframes\rotten_tomatoes_movies.csv")
criticsdf = pd.read_csv("dataframes\rotten_tomatoes_critic_reviews.csv")
critics = pd.read_pickle("dataframes\critics.pkl")

score_weight = 3
votes_weight = 0.01
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

#=======================================Definition of the functions========================================
def clean_title(a):
    return re.sub("[^a-zA-Z0-9 ]", "", a)

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = moviedf.iloc[indices].iloc[::-1]
    return results

# Transform the marks to numeric
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
    "F": 0,
}


def standardize(rev):  # some of them are /4 and some /5. Better eval?
    if "/" in str(rev):
        try:
            return eval(rev) * 100
            print(rev)
        except:
            print(rev)
            pass
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
    filt = moviedf.loc[moviedf["rotten_tomatoes_link"] == link, "movie_title"]
    if filt.empty:
        return None
    return filt.iloc[0]

#========================================== Little bit of cleaning ========================== 
criticsdf.dropna(subset="review_score", inplace=True)
criticsdf["review_score"] = criticsdf["review_score"].apply(standardize)
criticsdf.drop("review_content", axis=1, inplace=True)
criticsdf.drop("review_date", axis=1, inplace=True)
criticsdf.reset_index(inplace=True)

columns_titles = [
    "critic_name",
    "top_critic",
    "publisher_name",
    "rotten_tomatoes_link",
    "review_score",
    "review_type",
]
criticsdf = criticsdf.reindex(columns=columns_titles)
criticsdf.dropna(axis=0, inplace=True)
criticsdf["review_score"] = criticsdf["review_score"].round().astype(int)

#======================================= Streamlit ===================================================
# ======================== Presentation of the datas on the dashboard ================================
dashboard = st.sidebar.radio(
    "What dashboard do you want to see ?",
    (
        "A short presentation of the database",
        "The evolution of cinema over the years",
        "Hate",
        "Crush",
    ),
)

if dashboard == "A short presentation of the database":
    movcrit = st.selectbox(
        "Please choose the dataset you'd like to explore:  ", ("Movies", "Critics")
    )
    if movcrit == "Movies":
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        sns.histplot(
            data=df_titles["startYear"][df_titles["startYear"] != "\\N"].apply(
                lambda x: int(x)
            ),
            binwidth=5,
            ax=ax[0],
            color="red",
        )
        ax[0].set_title("Number of movies per year")
        ax[0].set_xlabel("Year")
        ax[0].set_ylabel("Number of movies")
        sns.histplot(
            data=df_titles, x="averageRating", binwidth=0.5, ax=ax[1], color="red"
        )
        ax[1].set_title("The distribution of ratings")
        ax[1].set_ylabel("Number of movies")
        top10 = pd.DataFrame(
            df_ov500.iloc[:, 16:].sum().sort_values(ascending=False).head(10)
        )
        sns.barplot(x=top10.index, y=top10.iloc[:, 0], color="red", ax=ax[2])
        ax[2].set_title("The 10 most represented genres")
        ax[2].set_ylabel("Number of movies")
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

        # Plot 1 - Top subplot (Seaborn Bar Plot)
        critic_reviews = (
            critics.groupby("critic_name")["indexcol"].count().reset_index()
        )

        bins = [0, 10, 20, 30, 40, 50, 100, 200, 500]
        labels = [
            "0-10",
            "11-20",
            "21-30",
            "31-40",
            "41-50",
            "51-100",
            "101-200",
            "201-500",
        ]
        critic_reviews["review_count_group"] = pd.cut(
            critics["indexcol"], bins=bins, labels=labels
        )

        sns.countplot(data=critic_reviews, x="review_count_group", ax=ax1, color="red")
        ax1.set_title("Number of Reviews Written by Film Critics")
        ax1.set_xlabel("Number of Reviews")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2 - Bottom subplot (Seaborn Bar Plot)
        review_sentiments = critics["review_type"].value_counts()

        sns.barplot(
            x=review_sentiments.index, y=review_sentiments.values, ax=ax2, color="red"
        )
        ax2.set_title("Number of Positive vs. Negative Reviews")
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Number of Reviews")

        fig.tight_layout()
        st.pyplot(fig)

elif dashboard == "The evolution of cinema over the years":
    st.title("The evolution of the cinema through the years")
    timeline = st.selectbox(
        "Pick what you'd like to see :  ", ("Charts", "Animated Charts")
    )
    if timeline == "Charts":
        # Lengthtime over years
        st.subheader("Evolution of the movie's length")
        df_titles_len = df_titles.dropna()
        df_titles_len = df_titles_len[df_titles_len["runtimeMinutes"] != "\\N"][
            df_titles_len["startYear"] != "\\N"
        ]
        mean_per_year = (
            df_titles_len[["startYear", "runtimeMinutes"]][
                df_titles_len["runtimeMinutes"] != "\\N"
            ][df_titles_len["startYear"] != "\\N"]
            .groupby(by=df_titles_len["startYear"])
            .median()
        )
        st.write(
            "Median of the movies before 1925 =",
            mean_per_year[mean_per_year["startYear"] < 1925]["runtimeMinutes"].median(),
            " minutes. Median of the movies after 2010 =",
            mean_per_year[mean_per_year["startYear"] < 2010]["runtimeMinutes"].median(),
            "minutes.",
        )
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=mean_per_year, x="startYear", y="runtimeMinutes", color="red"
        )
        ax.set_title("Median lengthtime regarding the year")
        ax.set_ylabel("Lengthtime in minutes")
        ax.set_xlabel("Year")
        st.pyplot(fig)

        # Ratings over years
        st.subheader("Evolution of the ratings")
        fig, ax = plt.subplots()
        mean_per_year = (
            df_ov500[["startYear", "averageRating"]][
                df_ov500["averageRating"] != "\\N"
            ][df_ov500["startYear"] != "\\N"]
            .groupby(by=df_ov500["startYear"])
            .mean()
        )
        sns.scatterplot(
            data=mean_per_year, x="startYear", y="averageRating", color="red"
        )
        ax.set_title("Average ratings regarding the year")
        ax.set_ylabel("Rate over 10")
        ax.set_xlabel("Year")
        st.pyplot(fig)
    else:
        video_file = open("EvolutionTOP10Genres.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        
#======================================= Start of the ML algorithms ======================================      
#================================= The algorithm based on a movie you like ===============================
elif dashboard == "Crush": 
    st.title("Pick a movie you have a crush on!")
    # Recommandation algorithm page
    movie_title = st.selectbox(
        "Please enter your favourite movie's title : ", (list(df["originalTitle"]))
    )
    num_sim = st.slider("How many similar movies do you want to see?", 1, 10, 5)
    database = st.radio(
        "Select the database to explore :", ("All", "Anglophone", "Non Anglophone")
    )

    if database == "Anglophone":
        columns_of_interest = ["isAdult", "runtimeMinutes", "averageRating"] + list(
            df.iloc[:, -26:-4].columns
        )
        X = df_US[columns_of_interest]
        distanceKNN = NearestNeighbors(n_neighbors=num_sim + 1).fit(X)
        coord = distanceKNN.kneighbors(
            df.loc[df["originalTitle"] == movie_title, columns_of_interest]
        )
        for i in range(1, num_sim + 1):
            st.write(
                "TOP",
                i,
                ":",
                df_US["originalTitle"].iloc[coord[1][0][i]],
                "(",
                str(df_US["startYear"].iloc[coord[1][0][i]]),
                ")",
            )
    elif database == "Non Anglophone":
        columns_of_interest = ["isAdult", "runtimeMinutes", "averageRating"] + list(
            df.iloc[:, -26:-4].columns
        )
        X = df_NoUS[columns_of_interest]
        distanceKNN = NearestNeighbors(n_neighbors=num_sim + 1).fit(X)
        coord = distanceKNN.kneighbors(
            df.loc[df["originalTitle"] == movie_title, columns_of_interest]
        )
        for i in range(1, num_sim + 1):
            st.write(
                "TOP",
                i,
                ":",
                df_NoUS["originalTitle"].iloc[coord[1][0][i]],
                "(",
                str(df_NoUS["startYear"].iloc[coord[1][0][i]]),
                ")",
            )
    else:
        columns_of_interest = ["isAdult", "runtimeMinutes", "averageRating"] + list(
            df.iloc[:, -26:-4].columns
        )
        X = df[columns_of_interest]
        distanceKNN = NearestNeighbors(n_neighbors=num_sim + 1).fit(X)
        coord = distanceKNN.kneighbors(
            df.loc[df["originalTitle"] == movie_title, columns_of_interest]
        )
        for i in range(1, num_sim + 1):
            st.write(
                "TOP",
                i,
                ":",
                df["originalTitle"].iloc[coord[1][0][i]],
                "(",
                str(df["startYear"].iloc[coord[1][0][i]]),
                ")",
            )

#============================= The ML algorithm based on a movie you hate =====================================

elif dashboard == "Hate":
    st.title("Pick a movie you hate!")
    movie_title = st.selectbox(
        "Please enter your least favourite movie's title : ",
        (list(moviedf["movie_title"])),
    )
    filtered_df = moviedf[moviedf["movie_title"] == movie_title]
    # Get the Rotten Tomatoes link from the filtered row
    mov = filtered_df["rotten_tomatoes_link"].values[0]

    moviehaters = criticsdf[
        (criticsdf["rotten_tomatoes_link"] == mov) & (criticsdf["review_score"] <= 2.5)
    ]
    # movieboys = criticdummies[(criticdummies['review_type'] == "Fresh")]
    crits = moviehaters["critic_name"].values.tolist()
    chosenones = criticsdf.loc[(criticsdf["critic_name"].isin(crits))]
    chosenones["review_score"] = chosenones["review_score"].apply(standardize2)
    average_scores = chosenones.groupby("rotten_tomatoes_link")["review_score"].mean()
    num_votes = chosenones.groupby("rotten_tomatoes_link")["review_type"].count()
    chosenflicks = pd.DataFrame(
        {"rotten_tomatoes_link": chosenones["rotten_tomatoes_link"].unique()}
    )
    chosenflicks = chosenflicks[chosenflicks.rotten_tomatoes_link != mov]
    chosenflicks["AverageScore"] = chosenflicks["rotten_tomatoes_link"].map(
        average_scores
    )
    chosenflicks = chosenflicks[chosenflicks.AverageScore > 3.99]
    chosenflicks["NumVotes"] = chosenflicks["rotten_tomatoes_link"].map(num_votes)
    chosenflicks["ObjectiveScore"] = chosenflicks.apply(
        lambda row: (score_weight * row["AverageScore"])
        + (votes_weight * row["NumVotes"])
        + random.uniform(-0.6, 0.6),
        axis=1,
    )
    # chosenflicks['rotten_tomatoes_link'] = chosenflicks['rotten_tomatoes_link'].apply(dubber)
    chosenflicks = chosenflicks.sort_values(["ObjectiveScore"], ascending=False)
    recs = chosenflicks.head(3)
    chosenflicks = chosenflicks.sort_values(["AverageScore"], ascending=False)
    recs = recs.append(chosenflicks.head(3), ignore_index=True)
    chosenflicks = chosenflicks.sort_values(["NumVotes"], ascending=False)
    recs = recs.append(chosenflicks.head(3), ignore_index=True)
    recs.drop(["ObjectiveScore", "NumVotes"], axis=1, inplace=True)
    recs.drop_duplicates(inplace=True, keep="first")
    recs = recs.sort_values(["AverageScore"], ascending=False)
    recs["rotten_tomatoes_link"] = recs["rotten_tomatoes_link"].apply(dubber)
    if recs.empty:
        st.write(
            "Looks like no critics hated this movie. You're probably either a diehard cinematophile or just insufferable. "
        )
    else:
        st.write("Critics who also hated this movie liked these movies: ")
        st.dataframe(data=recs, width=None, height=None, use_container_width=True)
