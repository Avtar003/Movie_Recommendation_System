import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df_columns = ['user_id','item_id','rating','timestamp']
df = pd.read_csv(r"D:\yadav\Coding Blocks\ml-100k\udata",sep="\t",names=df_columns)



movie_titles = pd.read_csv(r"D:\yadav\Coding Blocks\ml-100k\u.item",sep="|",encoding='latin-1',header=None)
movie_titles = movie_titles[[0,1]]
movie_titles.columns = ['item_id','title']

df = pd.merge(df,movie_titles,on='item_id')



ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['number of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
ratings.sort_values(by='rating',ascending=False)


sns.jointplot(x=ratings['rating'],y=ratings['number of ratings'])


movie_mat = df.pivot_table(index='user_id',columns='title',values='rating')
ratings.sort_values('number of ratings',ascending=False).head()
star_wars_rating = movie_mat['Star Wars (1977)']
similar_to_star_wars = movie_mat.corrwith(star_wars_rating)
corr_starwars = pd.DataFrame(similar_to_star_wars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_starwars = corr_starwars.join(ratings['number of ratings'])



def predict_movies(movie_name):
    movie_rating = movie_mat[movie_name]
    similar_to_movie = movie_mat.corrwith(movie_rating)
    corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['number of ratings'])
    corr_movie = corr_movie.sort_values(by='Correlation',ascending=False)
    predictions = corr_movie[corr_movie['number of ratings'] > 100].sort_values(by='Correlation',ascending=False)
    return predictions



