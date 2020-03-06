from flask import Flask
import jinja2
import os
from google.cloud import datastore
from google.cloud import bigquery
import tensorflow
import html2text

client = bigquery.Client()
h = html2text.HTML2Text()
h.ignore_links = True


def sanitize_html(payload):
    words = h.handle(payload).split(" ")[0:25]
    return " ".join(words) + "..."


def fetch_questions(limit=10, debug=False):
    QUERY_OBJ = (
            'SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` '
            'LIMIT %s' % limit)
    query_job = client.query(QUERY_OBJ)  # API request
    rows = query_job.result().to_dataframe()  # Waits for query to finish
    rows['body'] = rows['body'].apply(sanitize_html)
    rows['link'] = rows['id'].apply(lambda x: "/" + str(x))
    return rows.T.to_dict().values()


def fetch_question_by_id(question_id, debug=False):
    QUERY_OBJ = (
        'SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` '
        f'WHERE id = {question_id} '
        'LIMIT 1')
    query_job = client.query(QUERY_OBJ)  # API request
    row = query_job.result().to_dataframe()
    return list(row.T.to_dict().values())[0]


JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'])

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route("/", methods=["GET"])
def main_page():
    """Return a friendly HTTP greeting."""
    template = JINJA_ENVIRONMENT.get_template('templates/mainpage.html')
    template_vals = {"questions": fetch_questions(10)}
    return template.render(template_vals)


@app.route("/<question_id>", methods=["GET"])
def question(question_id, debug=False):
    question_row = fetch_question_by_id(question_id, debug)
    template = JINJA_ENVIRONMENT.get_template('templates/question.html')
    template_vals = {"title": question_row['title'], "body": question_row['body']}
    return template.render(template_vals)


if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.

    # question(34307193, debug=True)
    app.run(host="127.0.0.1", port=8080, debug=True)