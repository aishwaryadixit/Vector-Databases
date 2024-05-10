# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:21:32 2023

@author: aishwarya_dixit
"""

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import pandas as pd 
#Import the sentence transformer encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2') 

#Add the dataset

df = pd.read_csv('quora_200.csv')

# If you want to work with both columns together as a DataFrame:
ids_and_texts = df[['id', 'questions']]
#print (ids_and_texts)

#Define storage location
qdrant = QdrantClient(path="db1")

#Create a collection

qdrant.recreate_collection(
	collection_name="my_quora",
	vectors_config=models.VectorParams(
		size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by the used model
		distance=models.Distance.COSINE
	)
)
#convert dataframe to dictionary
data_points = df.to_dict(orient='records')
       

#insert records into the collection
qdrant.upload_records(
	collection_name="my_quora",
	records=[
		models.Record(
			id=idx,
			vector=encoder.encode(doc["questions"]).tolist(),
			payload=doc
		) for idx, doc in enumerate(data_points)
	]
)


query = 'How do I recover deleted messages in Facebook?'
#sample queries :
#'Life can be boring. What to do?'
#'How do I recover deleted messages in Facebook?'

#Ask the engine a question. Based on the data set we have enetered it will find 3 simlar questions 
#entered by other users that match closely with your question
hits = qdrant.search(
	collection_name="my_quora",
	query_vector=encoder.encode(query).tolist(),
	limit=3
)
for hit in hits:
	print(hit.payload, "score:", hit.score)
