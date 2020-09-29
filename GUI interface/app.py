import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#---------------------------------------
import time
import pandas as pd
import re

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#remove comment on first execution
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


import io
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model

#---------------------------------
def pattern_replacement(pattern, tweet, replacement):
    r = re.findall('"' + pattern + '[\w]*"', tweet)
    for i in r:
        tweet = re.sub(i, replacement, tweet)

    return tweet;


def preprocess_data(tweet):
    tweet = pattern_replacement('@', tweet, '')

    # .. Removing
    tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)

    # ...Removing non-alphabets from the tweet
    r = re.findall("[^A-Za-z]", tweet)
    for i in r:
        tweet = tweet.replace(i, " ")

    # ...Removing preposiitons, conjuctions and pronouns from the tweet
    words = nltk.word_tokenize(tweet)
    tags = nltk.pos_tag(words)
    '''
    PRP	Personal pronoun
    DT	Determiner
    CC	Coordinating conjunction
    IN	Preposition or subordinating conjunction
    PRP$	Possessive pronoun
    VBP	Verb, non-3rd person singular present
    NNP	Proper noun, singular
    VBZ	Verb, 3rd person singular present
    VB	Verb, base form
    MD	Modal
    RB	Adverb
    VBD	Verb, past tense
    WP	Wh-pronoun
    CD	Cardinal number
    WRB	Wh-adverb
    WDT	Wh-determiner
    '''

    # del_tags = ['PRP','DT','CC', 'IN', 'PRP$', 'VBP', 'NNP', 'VBZ', 'VB', 'MD', 'RB', 'VBD', 'WP', 'CD', 'WRB', 'WDT']
    del_tags = ['PRP', 'DT', 'CC', 'IN', 'PRP$', 'VBP', 'MD', 'WP', 'CD', 'WRB', 'WDT']

    new_tags = []
    for ord_pair in tags:
        if ord_pair[1] not in del_tags and len(ord_pair[0]) > 3:
            new_tags.append(ord_pair[0])

    # ...Removing Stopwords from the tweet
    new_tags = [w for w in new_tags if not w in stop_words]

    tweet = " ".join(new_tags)

    return tweet

#=====------------------------------


app = Flask(__name__)
model = load_model('CNN_best_weights.01-0.8165.hdf5')


with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

maxlength=25

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['tweet']
        # .value()
        # text = request.form(['tweet'])
        print(text)

        prepro = []
        prepro.append(preprocess_data(text))
        print(prepro)
        seq = tokenizer.texts_to_sequences(prepro)
        print(seq)
        pad_seq = pad_sequences(seq, maxlen=maxlength, padding='post')
        print(pad_seq)
        pred = model.predict(pad_seq)
        print(pred)
        if (pred > 0.5):
            # print('Informative')
            prediction = 'Informative '
        else:
            # print('Non Informative')
            prediction = 'Non Informative'
        print('--------------------------')

    return render_template('index.html', prediction_text='Tweet is  {}'.format(prediction))
''''
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)