import pandas
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#_____________________data_______________________#
print("Loading data, please wait...")
data_set = pandas.read_csv("https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv")
data_set_features = data_set[["Movie_Genre" , "Movie_Keywords" , "Movie_Tagline" , "Movie_Cast" , "Movie_Director"]].fillna('')


#____________________functions_____________________#
def recommend(fav_movie_name: str):
    '''This funtion takes in name of your fav movie and returns a list of recommended movies'''

    close_match = match(fav_movie_name)    
    index_of_Movie = data_set[data_set.Movie_Title == close_match]['Movie_ID'].values[0]
    x = find_x()

    similarity_score = cosine_similarity(x)
    recommendation_Score = list(enumerate (similarity_score[index_of_Movie]))
    sorted_similar_movies = sorted(recommendation_Score, key = lambda x:x[1], reverse = True)

    recommended_movies = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title = data_set[data_set.index == index]["Movie_Title"].values[0]
        if i<31 : recommended_movies.append(title)
        else : break
        i += 1

    return recommended_movies

def find_x():
    '''finds the value of x'''
    
    x = data_set_features["Movie_Genre"] + ' ' + data_set_features["Movie_Keywords"] + ' ' + data_set_features["Movie_Tagline"] + ' ' + data_set_features["Movie_Cast"] + ' ' + data_set_features["Movie_Director"]
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(x)
    return x

def match(movie_name: str):
    '''takes the name of a movie and returns the closest match to a movie in the database'''

    movie_names = data_set["Movie_Title"].tolist()
    close_match= difflib.get_close_matches(movie_name , movie_names)[0]
    print(f"closest match to {movie_name} is {close_match}")
    return close_match


#_____________main_____________#
fav_movie_name = input("Enter name of your favorite movie: ")

try:
    recommended_movies = recommend(fav_movie_name)
    print("These movies are recommended: ")
    for movie in recommended_movies : print(f"{recommended_movies.index(movie) + 1}. {movie}")

except IndexError : print("No movie found...")