# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv("https://raw.githubusercontent.com/naru94/Collaborative-Filtering-Recommendation-System/main/dataset/Movie_data.csv", sep=',', names=column_names)

# Load the movie information in a DataFrame
#Load the move information in a DataFrame:
column_names = {'item_id':'Movie_ID', 'title':'Movie_Title'}
movies_title_df = pd.read_csv("https://raw.githubusercontent.com/naru94/Collaborative-Filtering-Recommendation-System/main/dataset/Movie_Id_Titles.csv")
movies_title_df.rename(columns = {'item_id':'Movie_ID', 'title':'Movie_Title'}, inplace = True)
#Merge the DataFrames:
movies_df = pd.merge(movies_df,movies_title_df, on='Movie_ID')

# Explore the dataset
print(f"\nSize of the movie_df dataset is {movies_df.shape}")
print(f"{movies_df.User_ID.nunique()} users")
print(f"{movies_df.Movie_ID.nunique()} movies")

# Create the user-item rating matrix
ratings = np.zeros((movies_df.User_ID.nunique(), movies_df.Movie_ID.nunique()))
for row in movies_df.itertuples():
    ratings[row[1]-1, row[3]-1] = row[4]

# Explore the ratings dataset
print(f"\nShape of the ratings dataset is {ratings.shape}")
sparsity = np.count_nonzero(ratings) / np.prod(ratings.shape) * 100
print(f"Sparsity of the ratings dataset is {sparsity:.2f}%")

# Create the item-item similarity matrix
rating_cosine_similarity = cosine_similarity(ratings)
rating_cosine_similarity

# Define the recommendation function
# Provide Recommendations
def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    # Get the location of the actual user in the User-Items matrix
    # Use it to index the User similarity matrix
    user_similarities = X_user[user]
    # obtain the indices of the top k most similar users
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    # Obtain the mean ratings of those users for all movies
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    # Discard already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    # return recommendations - top similar users rated movies
    rec_movies_a=rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: 'Movie_ID'}, inplace=True)
    return rec_movies_a

def movie_recommender_run(user_Name):
    #Get ID from Name
    user_ID=movies_df.loc[movies_df['User_Names'] == user_Name].User_ID.values[0]
    #Call the function
    temp=movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec=temp.merge(movies_title_df, how='inner')
    return top_k_rec

# View the Provided Recommendations
#Converting the 2D array into a DataFrame as expected by the movie_recommender function
ratings_df=pd.DataFrame(ratings)

# Test the recommendation function
user_ID=12
movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
