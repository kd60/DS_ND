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

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# load data
engine = create_engine('sqlite:///titanic.db')
print("connected ")
df = pd.read_sql_table('titanic', engine)

# load model
model = joblib.load(r"C:\Users\atefi\Desktop\Project 2 - Copy\models\classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    pclass=['Pclass_1','Pclass_2','Pclass_3']
    genre_counts = df.groupby(['Pclass_1','Pclass_2','Pclass_3']).count()['Survived']
    genre_names = pclass

    
    
    category_counts=df.agg("max",axis=1)
    category_names = df.columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        
        
        
         {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution Of Titanic   ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Titanic - Column"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of count survived calsses',
                'yaxis': {
                    'title': "count survived in each class"
                },
                'xaxis': {
                    'title': "Pclass"
                }
            }
        }
       
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
    classification_labels = model.predict([query])[7]
    classification_results = dict(zip(df.columns[:, :-1], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
