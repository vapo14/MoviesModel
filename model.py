from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import pandas as pd
import numpy as np


def get_director(x):
    """
    Get the director's name from the crew feature. If director is not listed, return NaN
    """
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    """
    Returns the list top 3 elements or entire list; whichever is more.
    """
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


def clean_data(x: str) -> str:
    """
    Function to convert all strings to lower case and strip names of spaces
    """
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def create_soup(x):
    """
    Parse the stringified features into their corresponding python objects
    """
    soup = ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + \
        ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    return soup


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refactors DataFrame to be used in model
    """
    # Replace NaN with an empty string
    df['overview'] = df['overview'].fillna('')

    for feature in ['cast', 'crew', 'keywords', 'genres']:
        df[feature] = df[feature].apply(literal_eval)

    # Define new director, cast, genres and keywords features that are in a suitable form.
    df['director'] = df['crew'].apply(get_director)

    for feature in ['cast', 'keywords', 'genres']:
        df[feature] = df[feature].apply(get_list)

    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        df[feature] = df[feature].apply(clean_data)

    df['soup'] = df.apply(create_soup, axis=1)

    # Reset index of our main DataFrame and construct reverse mapping as before
    df = df.reset_index()
    return df


def create_cosine_sim(df: pd.DataFrame):
    """
    Creates cosine similarity model to be used
    """
    # Import CountVectorizer and create the count matrix
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim


def get_recommendations(df: pd.DataFrame, title: str, cosine_sim) -> pd.Series:
    """
    Main recommender function
    Args
        title: str with name of the movie. Must be exact.
        cosine_sim: linear kernel with similarity function
    """
    # Construct a reverse map of indices and movie titles
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    recommendations = df['title'].iloc[movie_indices]

    # Return the top 10 most similar movies
    return recommendations


def initialize_model(file1: str = './tmdb_5000_credits.csv', file2: str = './tmdb_5000_movies.csv') -> tuple:
    """
    Creates base dataframe and model to be returned to server
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1.columns = ['id', 'tittle', 'cast', 'crew']
    df2 = df2.merge(df1, on='id')

    main_df = normalize_df(df2)
    cosine_sim = create_cosine_sim(main_df)
    return main_df, cosine_sim


df, cosine_sim = initialize_model()


print(get_recommendations(df, 'The Dark Knight Rises', cosine_sim))
print(get_recommendations(df, 'Black November', cosine_sim))
