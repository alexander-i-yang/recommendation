from __future__ import absolute_import, division, print_function, unicode_literals

import math

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CHUNKSIZE = 500000

def debug_recommend(thingy):
    return "Hello: " + str(thingy)


def preprocess_rating_chunk(chunk):
    chunk[['userId', 'movieId', 'timestamp']] = chunk[['userId', 'movieId', 'timestamp']].astype('int32')
    chunk[['rating']] = chunk[['rating']].astype('float32')
    chunk_pivot = chunk.pivot(index='movieId', columns='userId', values='rating')
    for index, row in chunk_pivot.iterrows():
        print(list(zip(row,row.index)))
        #row = [(index, i) if not math.isnan(i) else pass for i in row]
        #for i in row:
        #    print("ree: ", i)
    return chunk


print("Reading data...")
raw_rating_chunks = pd.read_csv('ml-25m/ratings.csv', chunksize=CHUNKSIZE)
raw_movie_chunks = pd.read_csv('ml-25m/movies.csv', chunksize=CHUNKSIZE)
raw_genome_scores = pd.read_csv('ml-25m/genome-scores.csv', chunksize=CHUNKSIZE)
raw_genome_tags = pd.read_csv('ml-25m/genome-tags.csv', chunksize=CHUNKSIZE)

i = 0

print("Preprocessing raw rating chunks")
for raw_chunk in raw_rating_chunks:
    preprocessed_chunk = preprocess_rating_chunk(raw_chunk)
    if i == 0: preprocessed_ratings = preprocessed_chunk
    i += 1
    print(i)
    preprocessed_ratings = pd.merge(preprocessed_ratings, preprocessed_chunk, on="movieId")
