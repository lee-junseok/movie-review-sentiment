#!/usr/bin/env python
# coding: utf-8

#reading the file
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import spacy
from spacy import displacy
import string
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

nlp = spacy.load('en_core_web_sm')

def tokenizer_f(x):
    sent = nlp(x)
    token = []
    for i in (sent):
        if i.lemma_ == '-PRON-':
            token.append(i.lower_)
        elif not i.is_stop and not i.lemma_.lower() in punct:
            token.append(i.lemma_.lower())
    return ' '.join(token)

if __name__ == '__main__':
    running = True
    while running:
        X_test = input("Please give a review sentence: ")
        X_test = X_test.replace('<br /><br />', '')
        X_test = X_test.replace('..', '')
        X_test = X_test.replace('...', '')
        X_test = X_test.replace('....', '')

        punct = string.punctuation

        X_test = np.array([tokenizer_f(X_test)])

        # tokenizer = Tokenizer(num_words=10000,oov_token="<00v>")

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        X_test = tokenizer.texts_to_sequences(X_test)
        X_test = pad_sequences(X_test, maxlen=400)
        model = load_model('my_model.h5')
        pred = model.predict(X_test)
        score = pred[0][1]
        print("\nScore: ",score)
        if score:
            print('Positive!')
        else:
            print('Negative!')
        var = input("Test another review?(y/n): ")
        running = (var == 'y' or var=='yes')
