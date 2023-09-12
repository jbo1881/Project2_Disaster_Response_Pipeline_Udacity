import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///Disaster_Messages.db')
df = pd.read_sql_table('Disaster_Messages', engine)

# load model
model = joblib.load("classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals - Genre distribution
    genre_counts = df['genre'].value_counts()
    genre_names = genre_counts.index.tolist()
    
    # Create visuals - Genre distribution pie chart
    genre_distribution_graph = {
        'data': [
            {
                'type': 'pie',
                'labels': genre_names,
                'values': genre_counts,
                'hoverinfo': 'label+percent'
            }
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'height': 500
        }
    }
    
    # Extract data needed for the second visualization (example: word count by category)
    category_word_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_word_counts.index.tolist()
    
    # Create the second visualization (example: bar chart of word count by category)
    category_word_counts_graph = {
        'data': [
            {
                'type': 'bar',
                'x': category_names,
                'y': category_word_counts,
                'marker': {
                    'color': 'skyblue'
                }
            }
        ],
        'layout': {
            'title': 'Word Count by Category',
            'yaxis': {
                'title': 'Word Count'
            },
            'xaxis': {
                'title': 'Category'
            }
        }
    }
    
    graphs = [
        genre_distribution_graph,
        category_word_counts_graph  
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
