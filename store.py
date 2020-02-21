import math

from flask import Flask
import jinja2
import os
import pandas as pd
from google.cloud import datastore
import tensorflow

datastore_client = datastore.Client()
CHUNKSIZE = 50000
NUMCHUNKS = 500

def store_movie(name, rating, tags):
    entity = datastore.Entity(key=datastore_client.key('movie'))
    entity.update({
        'name': name,
        'rating': rating,
        'tags': tags
    })
    datastore_client.put(entity)


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
        if cur_row/num_rows > cur_milestone:
            cur_milestone += 0.1
            print("▓", end="")
        list_row = zip(row, row.index)
        res = [t for t in list_row if not any(isinstance(n, float) and math.isnan(n) for n in t)]
        #print(res)

    return chunk


def preprocess_avg_rating_chunk(chunk):
    chunk[['userId', 'movieId', 'timestamp']] = chunk[['userId', 'movieId', 'timestamp']].astype('int32')
    chunk[['rating']] = chunk[['rating']].astype('float32')
    ratings = pd.DataFrame(chunk.groupby('movieId')['rating'].mean())
    return ratings

def store_all_data():
    print("Reading data...")
    raw_rating_chunks = pd.read_csv('ml-25m/ratings.csv', chunksize=CHUNKSIZE)
    raw_movie_chunks = pd.read_csv('ml-25m/movies.csv', chunksize=CHUNKSIZE)
    raw_genome_scores = pd.read_csv('ml-25m/genome-scores.csv', chunksize=CHUNKSIZE)
    raw_genome_tags = pd.read_csv('ml-25m/genome-tags.csv', chunksize=CHUNKSIZE)
    i = 0
    cur_percent = 0.1
    print("Preprocessing raw rating chunks")
    print("┌──────────┐")
    print("└", end="")
    for raw_chunk in raw_rating_chunks:
        avg_rating_chunk = preprocess_avg_rating_chunk(raw_chunk)
        if i == 0:
            preprocessed_ratings = avg_rating_chunk
        if i/float(NUMCHUNKS) > cur_percent:
            cur_percent += 0.1
            print("─", end="")
        i += 1
        # TODO: delete the line below
        #if i == 5: break
        preprocessed_ratings = pd.merge(preprocessed_ratings, avg_rating_chunk, on="movieId")
        preprocessed_ratings['rating'] = preprocessed_ratings.mean(axis=1)
        preprocessed_ratings = preprocessed_ratings['rating']
    print("┘")
    print(preprocessed_ratings)

if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    store_movie("Test", 2.5, ["Action", "Adventure"])

    store_all_data()

    app.run(host="127.0.0.1", port=8080, debug=True)
























