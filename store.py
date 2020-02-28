import math
import re
import sys
from datetime import time

from flask import Flask
import jinja2
import os
import pandas as pd
from google.cloud import datastore as db
import tensorflow

datastore_client = db.Client()
CHUNKSIZE = 50000
NUMCHUNKS = 500


def store_movie(movieId, title, genres, year, rating):
    entity = db.Entity(key=datastore_client.key('movie'))
    entity.update({
        'movieId': movieId,
        'title': title,
        'genres': genres,
        'year': year,
        'rating': rating,
    })
    datastore_client.put(entity)


def store_movies(dataframe):
    for i, row in dataframe.iterrows():
        store_movie(row.movieId, row.title, row.genres, row.year, row.rating)


def fetch_movies(limit):
    query = datastore_client.query(kind='movie')
    query.order = ['-rating']

    times = query.fetch(limit=limit)

    return times


JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'])

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    """Return a friendly HTTP greeting."""
    template = JINJA_ENVIRONMENT.get_template('templates/datastore.html')
    template_vals = {"movies": fetch_movies(10)}
    return template.render(template_vals)


def preprocess_rating_chunk(chunk):
    chunk[['userId', 'movieId', 'timestamp']] = chunk[['userId', 'movieId', 'timestamp']].astype('int32')
    chunk[['rating']] = chunk[['rating']].astype('float32')
    chunk_pivot = chunk.pivot(index='movieId', columns='userId', values='rating')
    num_rows = float(len(chunk_pivot.index))
    cur_row = 0
    cur_milestone = 0.1
    for index, row in chunk_pivot.iterrows():
        cur_row += 1
        if cur_row / num_rows > cur_milestone:
            cur_milestone += 0.1
            # print("▓", end="")
        list_row = zip(row, row.index)
        res = [t for t in list_row if not any(isinstance(n, float) and math.isnan(n) for n in t)]

    return chunk


def preprocess_avg_rating_chunk(chunk):
    chunk[['userId', 'movieId', 'timestamp']] = chunk[['userId', 'movieId', 'timestamp']].astype('int32')
    chunk[['rating']] = chunk[['rating']].astype('float32')
    ratings = pd.DataFrame(chunk.groupby('movieId')['rating'].mean())
    return ratings


def preprocess_genome_chunk(chunk):
    ret = pd.DataFrame({'tags': [0]})
    print("sliced")
    for key, item in chunk.groupby(['movieId']):
        print(item.nlargest(5, 'relevance'))
        ret[key] = item.nlargest(5, 'relevance')
    print(ret)
    ret.drop("movieId")
    print("recent: ")
    print(ret)
    ret = ret.reset_index().set_index('movieId').drop('level_1', axis=1)
    print(ret)
    return ret


def merge_rating_chunk(merged_dataset, processed_chunk, debug=False):
    merged_dataset = pd.merge(merged_dataset, processed_chunk, on="movieId", how="outer")
    merged_dataset['rating'] = merged_dataset.mean(axis=1)
    merged_dataset = merged_dataset.drop(['rating_x', 'rating_y'], axis=1)
    if debug:
        print(merged_dataset)
    return merged_dataset


def merge_genome_chunk(merged_dataset, processed_chunk, debug=False):
    if debug:
        print("in merge, pre merge dastaset:")
        print(merged_dataset)
        print(processed_chunk)
    merged_dataset = merged_dataset.merge(processed_chunk, how="outer", on="movieId")
    merged_dataset["relevance"] = merged_dataset[["relevance_x", "relevance_y"]].max(axis=1)
    merged_dataset = merged_dataset.drop(["relevance_x", "relevance_y"], axis=1)
    merged_dataset["tagId"] = merged_dataset[["tagId_x", "tagId_y"]].max(axis=1)
    merged_dataset = merged_dataset.drop(["tagId_x", "tagId_y"], axis=1)
    if debug:
        print("merged dataset:")
        print(merged_dataset)
    return merged_dataset


def process_genome_merged(merged_dataset, debug=False):
    return merged_dataset.groupby("movieId") \
        .apply(lambda x: dict(zip(x.tagId, x.relevance))) \
        .to_frame().rename(columns={0: "tags"})


def preprocess_movie_chunk(chunk):
    def find_parens(st):
        ret = st[st.find("(") + 1:st.find(")")]
        return ret if str.isdigit(ret) else 0

    chunk['year'] = chunk["title"].apply(find_parens)
    chunk['title'] = chunk['title'].apply(lambda x: re.sub(r"\(.*\)", "", x))
    chunk['genres'] = chunk['genres'].apply(lambda x: [] if x == "(no genres listed)" else x.split("|"))
    return chunk


def merge_movie_chunk(merged, chunk, debug):
    return merged.append(chunk)


def process_chunk(
        chunks,
        numchunks,
        chunk_func,
        merge_func,
        msg="",
        debug=False,
        final_process=lambda x: x):
    i = 0
    cur_percent = 0.1
    merged_dataset = pd.DataFrame()
    sys.stdout.write("├─%s─┼" % msg)
    sys.stdout.flush()
    if debug: print()
    for raw_chunk in chunks:
        processed_chunk = chunk_func(raw_chunk)
        if debug:
            print("processed chunk:")
            print(processed_chunk)
        if i == 0:
            merged_dataset = processed_chunk
            if debug:
                print("first boi merged:")
                print(merged_dataset)
                print("First boi!")
        else:
            if debug:
                print(str(i) + " boi merged:")
                print(merged_dataset)
            merged_dataset = merge_func(merged_dataset, processed_chunk, debug)
            if debug:
                print("Merged dataset:")
                print(merged_dataset)
                print(merged_dataset.describe())
        while (i + 1) / float(numchunks) > cur_percent:
            cur_percent += 0.1
            sys.stdout.write("═")
            sys.stdout.flush()
        if debug and i == 2:
            break
        i += 1
    print("┤")
    return final_process(merged_dataset)


def store_all_data(debug=False):
    print("Reading data...")
    print("Done.")
    print("\nProcessing data...")
    raw_rating_chunks = pd.read_csv('ml-25m/ratings.csv', chunksize=CHUNKSIZE)
    raw_movie_chunks = pd.read_csv('ml-25m/movies.csv', chunksize=CHUNKSIZE)
    raw_genome_scores = pd.read_csv('ml-25m/genome-scores.csv', chunksize=CHUNKSIZE)
    # not a large database
    raw_genome_tags = pd.read_csv('ml-25m/genome-tags.csv')
    print("┌─────────────────────────────────┬──────────┐")
    print("│               Task              │ Progress │")
    print("├─────────────────────────────────┼──────────┤")
    preprocessed_ratings = process_chunk(
        chunks=raw_rating_chunks,
        numchunks=500,
        chunk_func=preprocess_avg_rating_chunk,
        merge_func=merge_rating_chunk,
        msg="Preprocessing raw rating chunks",
        debug=debug,
    )
    preprocessed_movies = process_chunk(
        chunks=raw_movie_chunks,
        numchunks=2,
        chunk_func=preprocess_movie_chunk,
        merge_func=merge_movie_chunk,
        msg="Preprocessing raw movie chunks─",
        debug=debug
    )
    print("└─────────────────────────────────┴──────────┘")
    if debug:
        print("Ratings Dataframe:")
        print(preprocessed_ratings)
        print("Movie Dataframe:")
        print(preprocessed_movies)
    preprocessed_dataframe = preprocessed_movies.merge(preprocessed_ratings, on='movieId')
    print("storing this data:")
    print(preprocessed_dataframe)
    store_movies(preprocessed_dataframe)


def delete_all_movies():
    # response = input("About to delete all movies. Continue? (Y/N) ")
    response = "Y"
    if response == "N":
        return
    else:
        while True:
            query = datastore_client.query(kind='movie')
            movies = query.fetch(limit=200)
            print(datastore_client.key('movie'))
            # datastore_client.delete()
            for i in movies:
                print(i)
            time.sleep(0.5)



if __name__ == "__main__":
    store_all_data(debug=False)
    # delete_all_movies()
    app.run(host="127.0.0.1", port=8080, debug=True)
