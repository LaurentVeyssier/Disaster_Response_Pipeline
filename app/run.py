import json, io, base64, re
import plotly
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify, send_file
from plotly.graph_objs import Bar
import joblib
#from sklearn.externals import joblib
from sqlalchemy import create_engine

stop_words = stopwords.words("english")

app = Flask(__name__)

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)


# load model
model = joblib.load("../models/classifier.pk")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    label_vectors = df.select_dtypes(include=['int64']).iloc[:,1:]
    label_counts = label_vectors.sum(axis=0).sort_values(ascending=False)
    labels_proportion = (label_counts/len(label_vectors)*100).round(1)
    label_names = label_vectors.columns.tolist()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=labels_proportion
                )
            ],

            'layout': {
                'title': 'Share of Classification Labels in dataset ("related" excluded)',
                'yaxis': {
                    'title': "percentage of labels in dataset"
                },
                'xaxis': {
                    'title': None
                }
            }
        },

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    counter=Counter()
    [counter.update(tokenize(m)) for m in df.message]
    
    cloud = WordCloud(  stopwords = stop_words, 
                        background_color = 'black',
                        width=2000, height=1000
                        ).generate_from_frequencies(frequencies=counter)
    #plt.title('Dataset worldcloud - Size propoerional to word frequency')
    plt.imshow(cloud, interpolation="bilinear")
    #plt.axis('off')
    img = io.BytesIO()
    cloud.to_image().save(img, 'PNG')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, plot_url=plot_url)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()