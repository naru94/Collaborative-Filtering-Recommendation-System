# Collaborative Filtering Recommendation System
Recommendation systems are widely used in a number of applications to give a personalized user experience. Collaborative filtering uses the information of other users or items in the system to filter out information. A user-based (user-user) collaborative filtering recommendation system is a memory-based approach that utilizes the users’ interactions with the system to find similar users and recommend them the items that similar users have liked.

## Project Overview
In this project, we'll use an IMDB movie dataset to create a recommendation system for the users using the scikit-learn library. Then we'll use the Streamlit library to build a simple recommender application.

## Libraries Used
We'll be using the following libraries for this project:

* pandas: To store and manage data
* numpy: To handle all the numerical values in the dataset
* sklearn: To create the recommendation system
* cosine_similarity from sklearn.metrics.pairwise: To create a cosine similarity matrix

You can install these libraries using pip. For example:
> pip install pandas numpy scikit-learn

## Dataset Used
The dataset for this project is a MovieLens dataset, of which we are using two files:

* Movie_data.csv: contains 100003 ratings on 1682 movies from 944 users.
* Movie_Id_Titles.csv: contains information about the movies.

> Note: The file Movie_data.csv has randomly generated usernames against each user ID via Python’s names package.

## Project Tasks
The project involves the following tasks:

1. Import all the required libraries.
2. Load Movie_data.csv and Movie_Id_Titles.csv in DataFrames.
3. Join the DataFrames on Movie_ID.
4. Explore the dataset to understand its dimensions, statistical summary, and the number of ratings given by each user.
5. Calculate the sparsity of the interaction matrix.
6. Use cosine similarity to find the similarity among users.
7. Create a function to give movie recommendations to a user by finding the k most similar users, their average rating of the movies, and the top 10 rated movies.
8. Create an application that receives a User ID and provides all the recommendations for that specific user. Use the movie recommendation function created in the Jupyter Notebook as a wrapper function.
9. Connect the recommendation system to a Streamlit application and display the recommendations for a selected user in the form of movie names.
10. Display the rating graph of each recommended movie in a 5x2 grid.

## Conclusion
This project will help you understand how to create a recommendation system and build an application using the Streamlit library. It involves tasks such as exploring the dataset, calculating the sparsity of the interaction matrix, finding the similarity among users using cosine similarity, and recommending movies to a user based on their similarity with other users.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.