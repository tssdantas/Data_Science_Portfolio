import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

df = pd.read_csv('movie_rating.csv')

users_ids = df.userID.unique()
item_ids = df.itemID.unique()
index_maxuser = users_ids.argmax()
index_maxid = users_ids.argmax()
nr_users = users_ids[index_maxuser]
nr_items = item_ids[index_maxid-1]

similarity_matrix = np.zeros((nr_users,nr_items))
print (' nr_users : ' + str(nr_users) + ' nr_items : ' + str(nr_items) )

index_df = int(0)
for i in range(nr_users):
    for j in range(nr_items):
        similarity_matrix[i,j] = df.iloc[index_df][2]
        index_df += 1
    #outside loop

item_similarity = pairwise_distances(similarity_matrix.T, metric='cosine')

# Top 3 similar items for item id 3
print ("Similar items for item id 3: \n", pd.DataFrame(item_similarity).loc[2,pd.DataFrame(item_similarity).loc[2,:] > 0].sort_values(ascending=False)[0:3])