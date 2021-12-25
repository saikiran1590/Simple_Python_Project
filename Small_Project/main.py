#import the relevant modules
import pandas as pd
import numpy as np
import math
import re
import seaborn as sns
from scipy.sparse import csr_matrix


data=pd.read_csv('combined_dataset.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
total_ratings_grouping=data.groupby('Rating')['Rating'].agg(['count'])
total_movies=data['Rating'].isnull().sum()
total_customers=data['Cust_Id'].nunique()
customers=total_customers-total_movies
ratings=data['Rating'].count()
total_null_values=pd.DataFrame(pd.isnull(data.Rating))
ratings_dataset=total_null_values[total_null_values['Rating']==True]

#print(ratings_dataset.shape)#shape method gives the number of columns & rows in a dataframe in the frmat of tuple(rows, columns)
null_values_dataset=ratings_dataset.reset_index()
#ZIP function in python will convert all the data into a tuple. It needs tuples as arguments
#for Example
#a=(1, 2, 3) & b=(2, 3, 4)
#print(zip(a,b)
movie_np=[]
movie_id=1
for i,j in zip(null_values_dataset['index'][1:],null_values_dataset['index'][:-1]):#zip((1,2,3,4,5,6,7),(0,1,2,3,4,5,6,7))
    replace=np.full([1, i-j-1], movie_id)# ([1,547],1),([1, 145],2).....
    movie_np=np.append(movie_np, replace)
    movie_id+=1

#last movie entries
last_entry_records=(len(data)-total_movies)-len(movie_np)
replace2=np.full([1, last_entry_records],movie_id)
movie_np=np.append(movie_np, replace2)

#now add a column named movie_id
data=data[pd.notnull(data["Rating"])]
data['Movie_Id']=movie_np.astype(int)
data['Cust_Id']=data['Cust_Id'].astype(int)

#now remove all the movies those have less ratings
#for that first count the total number of ratings for a movie
rating_summary=data.groupby('Movie_Id').agg(['count', 'mean'])
summary=rating_summary['Rating']
#map function is loop through an iterator
summary.index=summary.index.map(int)

#now crate a bench mark
mean=round(summary['count'].quantile(0.5))

drop_list=summary[summary['count']<mean].index

#now remove all the users that are inactive
#we need to count the number of ratings given by a customer
user_summary=data.groupby('Cust_Id').agg('count')
benchmark=round(user_summary['Rating'].quantile(0.9))
users_drop_list=user_summary[user_summary['Rating']<=benchmark].index


#Now remove all the movies and customers those are below the benchmark from the original table
data=data[~data['Movie_Id'].isin(drop_list)]
data=data[~data['Cust_Id'].isin(users_drop_list)]
data=data.reset_index()
del data['index']
print(data)

#now we will prepare our dataset for SVD and SVD takes only a matrix as an input
#so, first let's convert our dataset into a sparse matrix
#pivot table is used to summarize the data
df_p=pd.pivot_table(data=data, values='Rating', index='Cust_Id', columns='Movie_Id')
print(df_p.head(10))

#now read the movie-titles CSV file
movies_df=pd.read_csv('movie_titles.csv', encoding='ISO-8859-1', header=None, names=['movie_id', 'Year', 'Movie_name'], usecols=[0,1,2])
#now make the movie_Id as the index value of the data frame
movies_df.set_index('movie_id', inplace=True)
print(movies_df)
